"""write_excel: the worked examples its help shows and the catalog check runs.

Part of ``write_excel``; the leading underscore keeps it out of the tool catalog.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

import polars as pl

from ..authoring import SCRATCH_TOKEN, Ports, WorkedExample, scratch_path

_PLAIN_PATH: Final[str] = f"{SCRATCH_TOKEN}/pack.xlsx"
_LAID_OUT_PATH: Final[str] = f"{SCRATCH_TOKEN}/laid_out.xlsx"
_FROM_TEMPLATE_PATH: Final[str] = f"{SCRATCH_TOKEN}/quarter.xlsx"
_TEMPLATE_PATH: Final[str] = f"{SCRATCH_TOKEN}/template.xlsx"


@contextmanager
def _template_setup() -> Iterator[None]:
    """Put a small hand-made workbook in the example's scratch folder, for the template example."""
    import openpyxl

    book = openpyxl.Workbook()
    book.remove(book.active)
    notes = book.create_sheet("Notes")
    notes["A1"] = "Written by hand; the flow never names this tab."
    book.save(scratch_path(_TEMPLATE_PATH))
    book.close()
    yield


def plain_write_example() -> WorkedExample:
    return WorkedExample(
        scenario="write two tables to a new workbook, one tab each",
        configuration={
            "path": _PLAIN_PATH,
            "sheets": [
                {
                    "port": "figures",
                    "sheet": "Figures",
                    "formats": [{"type": "Float64", "format": "#,##0.00"}],
                },
                {"port": "detail", "sheet": "Detail"},
            ],
        },
        explanation="This step reads the two ports named in 'sheets', 'figures' and 'detail', and "
        "writes each to its own tab in that order. The Figures tab has one format rule, so every "
        "decimal column shows as money: 'amount' reads 10.25, and the cell keeps every digit. The "
        "Detail tab has no rules, so the workbook's own formatting applies.",
        inputs=Ports(
            {
                "figures": pl.DataFrame({"key": ["r1", "r2"], "amount": [10.25, -5.5]}),
                "detail": pl.DataFrame({"key": ["r1", "r2"], "code": ["A", "B"]}),
            }
        ),
        outputs=Ports(),
    )


def template_example() -> WorkedExample:
    return WorkedExample(
        scenario="refresh one tab of a workbook someone built by hand",
        configuration={
            "path": _FROM_TEMPLATE_PATH,
            "template": _TEMPLATE_PATH,
            "sheets": [{"port": "figures", "sheet": "Figures"}],
        },
        explanation="The template already has a Notes tab. The Figures tab is written into a copy of "
        "the template, saved to 'path', so Notes comes through unchanged and the template itself is "
        "untouched for the next run.",
        inputs=Ports({"figures": pl.DataFrame({"key": ["r1", "r2"], "amount": [10.25, -5.5]})}),
        outputs=Ports(),
        setup=_template_setup,
    )


def laid_out_example() -> WorkedExample:
    return WorkedExample(
        scenario="lay out one tab as a filterable table and another as a title over paragraphs",
        configuration={
            "path": _LAID_OUT_PATH,
            "sheets": [
                {
                    "port": "figures",
                    "sheet": "Figures",
                    "formats": [{"type": "Float64", "format": "#,##0.00"}],
                    "fit_widths": True,
                    "table": {"style": "TableStyleLight1"},
                    "header": {"bold": True},
                },
                {
                    "port": "notes",
                    "sheet": "Notes",
                    "widths": [{"type": "String", "width": 100}],
                    "wrap": [{"type": "String"}],
                    "header": {"bold": True, "size": 14},
                },
            ],
        },
        explanation="The Figures tab becomes a table called 'Figures', with filter dropdowns on bold "
        "headings, and each column fits its heading or its longest figure as shown. The Notes tab has "
        "one column 100 characters wide, a bold 14-point heading as its title, and wrapped paragraphs. "
        "The values and number formats are the same as without these settings.",
        inputs=Ports(
            {
                "figures": pl.DataFrame({"key": ["r1", "r2"], "amount": [1234567.0, -5.5]}),
                "notes": pl.DataFrame({"About these figures": ["The first paragraph.", "The second paragraph."]}),
            }
        ),
        outputs=Ports(),
    )
