"""write_excel: how a written tab looks: headings, widths, wrapping and tables.

Layout runs after the cells are written and never changes a value or a number format.
"""

import re
from collections.abc import Mapping, Sequence
from copy import copy
from typing import Any, Final

import polars as pl

from ._workbook_checks import table_name
from ._workbook_rules import column_formats, format_rules, resolve_rules

#: The widest a fitted column gets, so a paragraph does not make a column hundreds wide.
_FIT_CAP: Final[float] = 60
#: Room either side of the text in a fitted column.
_FIT_PADDING: Final[float] = 2
#: Room for the filter button a table puts on each heading.
_FILTER_BUTTON: Final[float] = 2
#: The font size a heading is measured at when nobody states one.
_DEFAULT_FONT_SIZE: Final[float] = 11
#: The longest a number is shown under the spreadsheet's General format.
_GENERAL_NUMBER_LENGTH: Final[int] = 11


def lay_out_tab(worksheet: Any, frame: pl.DataFrame, entry: Mapping[str, Any]) -> None:
    """Apply a sheet entry's layout settings to a tab whose values are already written.

    Each step sets a font, a width, an alignment or a table, never a value or a number format.
    `TestLayoutNeverTouchesAFigure` fails if that stops being true.
    """
    header = entry.get("header")
    if header is not None:
        _style_headings(worksheet, frame.width, header)
    _set_widths(worksheet, frame, entry)
    _wrap_columns(worksheet, frame, entry.get("wrap") or ())
    if entry.get("table") is not None:
        _add_table(worksheet, frame, entry)


def _style_headings(worksheet: Any, width: int, header: Mapping[str, Any]) -> None:
    """Make the heading row bold, a stated size, or both. A setting not stated keeps the cell's font."""
    for index in range(1, width + 1):
        cell = worksheet.cell(row=1, column=index)
        font = copy(cell.font)
        if header.get("bold") is not None:
            font.b = header["bold"]
        if header.get("size") is not None:
            font.sz = header["size"]
        cell.font = font


def _set_widths(worksheet: Any, frame: pl.DataFrame, entry: Mapping[str, Any]) -> None:
    """Set each column's width: fitted first if asked for, then any stated width on top."""
    from openpyxl.utils import get_column_letter

    widths = _fitted_widths(frame, entry) if entry.get("fit_widths") else {}
    widths.update(resolve_rules(frame.schema, entry.get("widths") or (), "width"))
    for name, width in widths.items():
        letter = get_column_letter(frame.columns.index(name) + 1)
        worksheet.column_dimensions[letter].width = width


def _fitted_widths(frame: pl.DataFrame, entry: Mapping[str, Any]) -> dict[str, float]:
    """The width 'fit_widths' gives each column.

    A column is as wide as its heading or its longest value as shown, whichever is longer. Then it
    gets room either side, and it is never wider than the cap.

    The heading is measured at its own font size, so a 14-point heading takes more room than an
    11-point one. On a table, each heading also needs room for its filter button.
    """
    formats = column_formats(frame.schema, format_rules(entry))
    heading_size = (entry.get("header") or {}).get("size") or _DEFAULT_FONT_SIZE
    button = _FILTER_BUTTON if entry.get("table") is not None else 0

    widths: dict[str, float] = {}
    for name in frame.columns:
        heading = len(name) * heading_size / _DEFAULT_FONT_SIZE + button
        longest = max(heading, _longest_shown(frame[name], formats[name]))
        widths[name] = min(longest + _FIT_PADDING, _FIT_CAP)
    return widths


def _wrap_columns(worksheet: Any, frame: pl.DataFrame, rules: Sequence[Mapping[str, Any]]) -> None:
    """Wrap the data cells of each column the rules reach, and sit them at the top of the row.

    Every data cell is set, empty ones included. The heading is left alone.
    """
    from openpyxl.styles import Alignment

    wrapped = Alignment(wrap_text=True, vertical="top")
    for name in resolve_rules(frame.schema, rules):
        column = frame.columns.index(name) + 1
        for row in range(2, frame.height + 2):
            worksheet.cell(row=row, column=column).alignment = wrapped


def _add_table(worksheet: Any, frame: pl.DataFrame, entry: Mapping[str, Any]) -> None:
    """Make the tab's heading and rows a spreadsheet table, plain unless a built-in style is named.

    A port with no rows still gets a table: its heading and one empty row, the smallest table a
    spreadsheet accepts.
    """
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    last_row = max(frame.height + 1, 2)
    ref = f"A1:{get_column_letter(frame.width)}{last_row}"
    style = entry["table"].get("style")
    info = None if style is None else TableStyleInfo(name=style, showRowStripes=True)
    worksheet.add_table(Table(displayName=table_name(entry), ref=ref, tableStyleInfo=info))


def _longest_shown(column: pl.Series, code: str | None) -> float:
    """How many characters the longest value of ``column`` takes on screen, under ``code``.

    A figure is measured as it is shown, not as it is stored: 1234567.0 under '#,##0.00' shows as
    '1,234,567.00', and a column fitted to the stored digits shows '####'. Read from the column's
    extremes, not row by row.
    """
    values = column.drop_nulls()
    if values.is_empty():
        return 0
    dtype = column.dtype
    if dtype == pl.String:
        return float(values.str.len_chars().max() or 0)
    if dtype == pl.Boolean:
        return len("FALSE")
    if dtype.is_temporal():
        return _date_shown_length(code or "")
    if dtype.is_numeric():
        if dtype.is_float():
            values = values.filter(values.is_finite())
            if values.is_empty():
                return _GENERAL_NUMBER_LENGTH
        if code is None:
            return min(float(values.cast(pl.String).str.len_chars().max() or 0), _GENERAL_NUMBER_LENGTH)
        largest = float(values.abs().max())
        negative = float(values.min()) < 0
        return _number_shown_length(largest, negative, code)
    return float(values.cast(pl.String).str.len_chars().max() or 0)


def _number_shown_length(largest: float, negative: bool, code: str) -> int:
    """How long ``largest`` is shown under a number format code, erring on the wide side.

    Reads the decimals, the thousands separator and a percent sign from the code, and counts every
    other character in it as shown. That over-counts quotes and escapes, which only makes a column a
    little wider.
    """
    sections = code.split(";")
    section = re.sub(r"\[[^\]]*\]", "", sections[0])
    integer_part, _, decimal_part = section.partition(".")
    decimals = sum(1 for character in decimal_part if character in "0#?")
    if "%" in section:
        largest *= 100
    separator = "," if "," in integer_part else ""
    shown = len(f"{largest:{separator}.{decimals}f}")
    literal = sum(1 for character in section if character not in "0#?.,%")
    if negative:
        negative_section = re.sub(r"\[[^\]]*\]", "", sections[1]) if len(sections) > 1 else section
        literal = max(literal, sum(1 for character in negative_section if character not in "0#?.,%")) + 1
    return shown + literal + ("%" in section)


def _date_shown_length(code: str) -> int:
    """How long a date is shown under a date format code: the code's own length, with a month or
    day written out in full counted at its longest."""
    shown = re.sub(r'\[[^\]]*\]|["\\]', "", code)
    return len(re.sub(r"(?i)mmmm|dddd", "x" * len("September"), shown))
