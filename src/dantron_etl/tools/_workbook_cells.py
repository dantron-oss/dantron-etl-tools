"""write_excel: writing one frame's values and number formats into one tab.

Part of ``write_excel``; the leading underscore keeps it out of the tool catalog.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from ._workbook_rules import column_formats


def write_tab(worksheet: Any, frame: pl.DataFrame, rules: Sequence[Mapping[str, Any]]) -> None:
    """Write one frame to one tab: a header row, the rows under it, and the number formats.

    A date and an instant carry a format whether the tab asked for one or not, because that format
    is the only thing telling a cell it holds a date rather than a count of days. Everything else is
    formatted only where a rule says so: a format this tool chose for itself would be a claim about
    how the pack is read that it is not entitled to make.

    The header is written before the formats are worked out and is never given one. A heading is a
    word, and a money format on a word shows nothing while saying the writer cannot tell a heading
    from a figure.

    A cell with nothing in it is formatted like the rest of its column. The format belongs to the
    cell rather than to the value that happens to be in it: a money column with a gap in it is still
    a money column, and formatting around the gaps leaves a chequerboard that shows the moment
    anybody types into the file.
    """
    for index, name in enumerate(frame.columns, start=1):
        _as_text_if_formula_like(worksheet.cell(row=1, column=index, value=name))

    formats = column_formats(frame.schema, rules)
    # Once per column, in the order the cells are written. The loop under this one runs for every
    # cell of the pack, and a tab holds up to a million rows, so a lookup by name in there is a
    # million lookups for an answer that cannot change between rows.
    codes = tuple(formats[name] for name in frame.columns)

    for row_number, row in enumerate(frame.iter_rows(), start=2):
        for index, (value, code) in enumerate(zip(row, codes, strict=True), start=1):
            if value == "":
                value = None
            cell = worksheet.cell(row=row_number, column=index, value=value)
            if isinstance(value, str):
                _as_text_if_formula_like(cell)
            if code is not None:
                cell.number_format = code


def _as_text_if_formula_like(cell: Any) -> None:
    """Keep a text value that starts with ``=`` as text, not a formula.

    openpyxl stores any string starting with ``=`` as a formula, so a free-text value such as
    ``=HYPERLINK(...)`` would run on the reader's machine. The cell is forced back to text, and the
    quote prefix stops a spreadsheet program reading it as a formula when the cell is edited.
    """
    if cell.data_type == "f":
        cell.data_type = "s"
        cell.quotePrefix = True
