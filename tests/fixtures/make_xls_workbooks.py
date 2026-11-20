"""Build ``read_excel_1900.xls`` and ``read_excel_1904.xls``, the old-format workbooks read_excel's tests read.

The two files hold the same sheets. They differ only in the date system the workbook uses: Excel
for Windows counts days from 1900, old Excel for Mac counted from 1904, and an .xls file says which
in its header. ``SHEETS`` below is the content. The tests build an .xlsx from the same ``SHEETS``
and require read_excel to read both formats the same way.

The package does not depend on anything that writes .xls, so the files are committed. To rebuild
them, install ``xlwt`` (1.3.0 was used) into a scratch environment, never the project's, and run
this script from the repo root with that environment's Python:

    uv venv /tmp/xls-env && VIRTUAL_ENV=/tmp/xls-env uv pip install xlwt==1.3.0
    /tmp/xls-env/bin/python tests/fixtures/make_xls_workbooks.py

Only ``build`` imports xlwt, so the tests can import ``SHEETS`` without it.

Three things xlwt cannot write, so no sheet here holds them:

- Negative zero. xlwt stores a whole number in its short form, which has no negative zero.
- A text cell holding nothing. xlwt writes an empty string as a blank cell.
- A formula with its last result. xlwt writes the formula alone, so an error value is written as a
  plain error cell instead.

The values are invented.
"""

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
TARGETS = {False: HERE / "read_excel_1900.xls", True: HERE / "read_excel_1904.xls"}


@dataclass(frozen=True)
class Shown:
    """A cell stored as ``value`` and shown with Excel's number format ``num_format``."""

    value: Any
    num_format: str


@dataclass(frozen=True)
class ErrorValue:
    """A cell holding an error value, such as #N/A, as a formula leaves behind."""

    code: str


@dataclass(frozen=True)
class Merged:
    """``text`` in a cell merged across ``width`` columns, starting at this one."""

    text: str
    width: int


#: The number format each kind of cell is written with unless it carries its own.
DATE_FORMAT = "yyyy-mm-dd"
STAMP_FORMAT = "yyyy-mm-dd hh:mm:ss"
TIME_FORMAT = "hh:mm"
#: Excel's own short date format, number 14, which a file from Excel uses most.
BUILT_IN_DATE_FORMAT = "M/D/YY"

#: The code xlwt needs for each error value.
ERROR_CODES = {"#N/A": 0x2A, "#DIV/0!": 0x07, "#VALUE!": 0x0F}

_CELLS = [
    12,
    12.0,
    -40,
    123456789012345,
    0.1,
    10.25,
    99999999999.99,
    1 / 3,
    1e20,
    2.5e-10,
    True,
    False,
    dt.date(2026, 1, 31),
    dt.datetime(2026, 1, 31, 14, 5, 9),
    dt.datetime(2026, 1, 31, 0, 0, 0),
    dt.time(14, 5),
    Shown(dt.date(2026, 2, 28), BUILT_IN_DATE_FORMAT),
    Shown(7, "000"),
    "007",
    "3600",
    "12.50",
    None,
    "n/a",
    "café €",
    "   ",
]

#: Every sheet of both workbooks, row by row, in workbook order. None is a blank cell.
SHEETS: dict[str, list[list[Any]]] = {
    "Cells": [["value", "key"], *([cell, f"r{index}"] for index, cell in enumerate(_CELLS, start=1))],
    "Typed": [
        ["figure", "whole", "on", "shown", "at", "flag", "text"],
        [
            10.25,
            10,
            dt.date(2026, 1, 31),
            Shown(dt.date(2026, 1, 31), BUILT_IN_DATE_FORMAT),
            dt.datetime(2026, 1, 31, 14, 5, 9),
            True,
            "007",
        ],
        [None, None, None, None, None, None, None],
        [
            -5.5,
            20,
            dt.date(1999, 12, 31),
            Shown(dt.date(1999, 12, 31), BUILT_IN_DATE_FORMAT),
            dt.datetime(2026, 2, 1, 9, 30),
            False,
            "a",
        ],
    ],
    "Titled": [["Orders for the period", None], [None, None], ["key", "amount"], ["o1", 10.25], ["o2", 2026]],
    "Long": [["key", "amount"], *([f"o{i}", i + 0.25] for i in range(1, 151)), ["o151", "n/a"]],
    "Missing": [["key", "amount"], ["o1", 1.5], ["o2", 2.5], ["o3", "#N/A"]],
    "Errors": [["value", "key"], *([ErrorValue(code), f"r{i}"] for i, code in enumerate(ERROR_CODES, start=1))],
    "Merged": [[Merged("order", 2), None, "amount"], ["o1", "A", 10.25]],
    "Headings": [["key", 2026, 1.5], ["o1", 10.25, 3]],
    "Cover": [["Monthly orders"]],
    "Jan 2026": [["key", "amount"], ["j1", 10], ["j2", 4]],
    "Feb 2026": [["key", "amount"], ["f1", 2.5]],
    "Empty": [],
}


def build(dates_1904: bool) -> Path:
    """Write one of the two workbooks with xlwt, and return its path."""
    import xlwt

    book = xlwt.Workbook()
    book.dates_1904 = dates_1904
    styles: dict[str, Any] = {}

    def style(num_format: str) -> Any:
        return styles.setdefault(num_format, xlwt.easyxf(num_format_str=num_format))

    for name, rows in SHEETS.items():
        sheet = book.add_sheet(name)
        for row_index, row in enumerate(rows):
            for column_index, cell in enumerate(row):
                value, num_format = (cell.value, cell.num_format) if isinstance(cell, Shown) else (cell, None)
                if isinstance(value, ErrorValue):
                    sheet.row(row_index).set_cell_error(column_index, ERROR_CODES[value.code])
                elif isinstance(value, Merged):
                    last = column_index + value.width - 1
                    sheet.write_merge(row_index, row_index, column_index, last, value.text)
                elif value is None:
                    continue
                elif isinstance(value, dt.datetime):
                    sheet.write(row_index, column_index, value, style(num_format or STAMP_FORMAT))
                elif isinstance(value, dt.date):
                    sheet.write(row_index, column_index, value, style(num_format or DATE_FORMAT))
                elif isinstance(value, dt.time):
                    sheet.write(row_index, column_index, value, style(num_format or TIME_FORMAT))
                elif num_format is not None:
                    sheet.write(row_index, column_index, value, style(num_format))
                else:
                    sheet.write(row_index, column_index, value)
    target = TARGETS[dates_1904]
    book.save(str(target))
    return target


if __name__ == "__main__":
    for dates_1904 in (False, True):
        built = build(dates_1904)
        print(f"{built} -- {built.stat().st_size} bytes")
