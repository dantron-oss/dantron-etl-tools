"""write_excel: writing one frame's values and number formats into one tab.

Part of ``write_excel``; the leading underscore keeps it out of the tool catalog.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from ._workbook_rules import column_formats


def write_tab(worksheet: Any, frame: pl.DataFrame, rules: Sequence[Mapping[str, Any]]) -> None:
    """Write one frame to one tab: a header row, the rows under it, and the number formats.

    Dates and timestamps always get a date format; other columns only where a rule says so. Headings
    never get a format. An empty cell is formatted like the rest of its column.
    """
    for index, name in enumerate(frame.columns, start=1):
        _as_text_if_formula_like(worksheet.cell(row=1, column=index, value=name))

    formats = column_formats(frame.schema, rules)
    # Looked up once per column, not once per cell.
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
    """Keep a text value that starts with `=` as text, not a formula.

    openpyxl stores such a string as a formula, so `=HYPERLINK(...)` would run on the reader's machine.
    The quote prefix stops the spreadsheet reading it as a formula when the cell is edited.
    """
    if cell.data_type == "f":
        cell.data_type = "s"
        cell.quotePrefix = True
