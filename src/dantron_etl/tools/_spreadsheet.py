"""What a spreadsheet cell and tab can hold. Shared by the tools that write a workbook.

These are the destination's rules, not any one tool's. Every tool that writes a workbook reads this
one copy, so two tools cannot drift into two different rules.

The leading underscore keeps this module out of the tool catalog: ``tools/__init__.py`` imports
every module in this folder except the ones whose name starts with ``_``.
"""

from typing import Final

import polars as pl

from ..authoring import ConfigError

# A worksheet's own row limit, published so a refusal about it can be checked against a number
# rather than taken on faith, and so a caller building a definition file can read it too.
MAX_SHEET_ROWS: Final[int] = 1_048_576
# One row of the sheet is always the header, so this many are left for data.
MAX_DATA_ROWS: Final[int] = MAX_SHEET_ROWS - 1
# The largest whole number a double can still tell apart from its neighbour. 2**53 itself is
# representable; its successor collapses onto it, which is why this is the boundary rather than
# 2**53 or some rounder-looking number nearby.
MAX_EXACT_INTEGER: Final[int] = 2**53 - 1

# The most characters one cell holds. A longer text would be cut to this length without a word.
MAX_CELL_TEXT: Final[int] = 32_767

# What a workbook lets a tab be called. Published alongside the limits above, and for the same
# reason. These are the destination's rules rather than this tool's, and every tool in the catalog
# that writes a workbook reads this one copy. Two copies would eventually be two different rules.
FORBIDDEN_SHEET_CHARACTERS: Final[str] = "[]:*?/\\"
MAX_SHEET_NAME_LENGTH: Final[int] = 31


def require_xlsx_path(tool: str, path: str) -> None:
    """Refuse a ``path`` that does not end in ``.xlsx``, in the words of the tool that was given it."""
    if not path.lower().endswith(".xlsx"):
        raise ConfigError(f"{tool} only writes .xlsx workbooks; got path '{path}'")


def unsupported_dtype_reason(dtype: pl.DataType) -> str | None:
    """Why ``dtype`` cannot land in a cell honestly, or ``None`` if it can.

    Each of these has a plausible-looking wrong answer available -- which is exactly why none of
    them is allowed through silently.

    Public because the answer is a property of the destination rather than of this tool. Every tool
    in the catalog that writes a workbook is bound by it and reads this one copy. Two copies of the
    answer would eventually be two different answers.
    """
    if isinstance(dtype, pl.Decimal):
        return "a Decimal column's whole purpose is an exactness a cell does not keep"
    if isinstance(dtype, pl.Time):
        return "a cell has no notion of a time of day on its own; it would arrive as a date in 1899"
    if isinstance(dtype, pl.Duration):
        return "a cell has no notion of a length of time; it would arrive as a fraction of a day"
    if isinstance(dtype, pl.List):
        return "a cell holds one value, not a list; it would arrive as a line of punctuation"
    if isinstance(dtype, pl.Array):
        # An Array is a list of a fixed length. Polars does not make it a kind of List, so it has to
        # be named here as well, or a fixed-size list column walks past this check.
        return "a cell holds one value, not a list; it would arrive as a line of punctuation"
    if isinstance(dtype, pl.Binary):
        return (
            "a cell holds text or a number, not raw bytes; bytes that happen to decode would "
            "arrive as words and look like data somebody meant to send"
        )
    if isinstance(dtype, pl.Struct):
        return "a cell holds one value, not a nested record"
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        return "a cell cannot record a time zone offset alongside an instant"
    return None
