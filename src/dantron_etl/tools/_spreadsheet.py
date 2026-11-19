"""What a spreadsheet cell and tab can hold, shared by the tools that write a workbook.

These are the spreadsheet's limits, kept in one place so two tools cannot disagree about them.
A module whose name starts with `_` is a helper, not a tool, and is left out of the catalog.
"""

from typing import Final

import polars as pl

from ..authoring import ConfigError

# A worksheet's row limit.
MAX_SHEET_ROWS: Final[int] = 1_048_576
# One row of the sheet is always the header, so this many are left for data.
MAX_DATA_ROWS: Final[int] = MAX_SHEET_ROWS - 1
# The largest whole number a cell (a double) holds exactly, with its neighbour still distinct.
MAX_EXACT_INTEGER: Final[int] = 2**53 - 1

# The most characters one cell holds. A longer text would be cut to this length without a word.
MAX_CELL_TEXT: Final[int] = 32_767

# What a tab may be called.
FORBIDDEN_SHEET_CHARACTERS: Final[str] = "[]:*?/\\"
MAX_SHEET_NAME_LENGTH: Final[int] = 31


def require_xlsx_path(tool: str, path: str) -> None:
    """Refuse a ``path`` that does not end in ``.xlsx``, in the words of the tool that was given it."""
    if not path.lower().endswith(".xlsx"):
        raise ConfigError(f"{tool} only writes .xlsx workbooks; got path '{path}'")


def unsupported_dtype_reason(dtype: pl.DataType) -> str | None:
    """Why a column of `dtype` cannot be written to a cell faithfully, or None if it can."""
    if isinstance(dtype, pl.Decimal):
        return "a Decimal column's whole purpose is an exactness a cell does not keep"
    if isinstance(dtype, pl.Time):
        return "a cell has no notion of a time of day on its own; it would arrive as a date in 1899"
    if isinstance(dtype, pl.Duration):
        return "a cell has no notion of a length of time; it would arrive as a fraction of a day"
    if isinstance(dtype, pl.List):
        return "a cell holds one value, not a list; it would arrive as a line of punctuation"
    if isinstance(dtype, pl.Array):
        # A fixed-length list. polars does not treat it as a List, so it is named separately.
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
