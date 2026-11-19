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
    """A curated workbook already on disk, standing in for the pack somebody built by hand.

    Planted with :func:`scratch_path`, which resolves the same ``{scratch}`` token the example's own
    configuration holds, into the directory :func:`check_tool` opens for the run. Nothing here tears
    the file back down: the directory is the checker's, and it takes whatever a setup left inside it.
    """
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
        scenario="write a pack of two tabs where no workbook exists yet",
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
        explanation="The two entries in 'sheets' are what say this step reads 'figures' and "
        "'detail'. Each frame lands on its own tab, in the order the entries name them, and "
        "nothing leaves on an output port. The first tab states one format rule, so every decimal "
        "column on it is shown as money -- 'amount' reads 10.25 and would read 1,234.57 for a "
        "figure that long, while the cell keeps every digit that went in. The second tab states "
        "none, so its cells are shown however the workbook shows them.",
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
        scenario="refresh one tab of a workbook somebody built by hand",
        configuration={
            "path": _FROM_TEMPLATE_PATH,
            "template": _TEMPLATE_PATH,
            "sheets": [{"port": "figures", "sheet": "Figures"}],
        },
        explanation="The template already holds a 'Notes' tab. It is loaded, the 'Figures' tab is "
        "written into it, and the whole workbook is saved to 'path' -- so 'Notes' arrives in the "
        "pack exactly as it was, and the template itself is untouched and ready for the next run.",
        inputs=Ports({"figures": pl.DataFrame({"key": ["r1", "r2"], "amount": [10.25, -5.5]})}),
        outputs=Ports(),
        setup=_template_setup,
    )


def laid_out_example() -> WorkedExample:
    return WorkedExample(
        scenario="lay out a figures tab as a table and a notes tab as a title over paragraphs",
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
        explanation="The 'Figures' tab becomes a table called 'Figures', with filter dropdowns on "
        "its bold headings, and each column is as wide as its heading or its longest figure as "
        "shown. The 'Notes' tab has one column 100 characters wide, a bold 14-point heading that "
        "reads as its title, and paragraphs that wrap. Every value and number format is the same "
        "as it would be without these options.",
        inputs=Ports(
            {
                "figures": pl.DataFrame({"key": ["r1", "r2"], "amount": [1234567.0, -5.5]}),
                "notes": pl.DataFrame({"About these figures": ["The first paragraph.", "The second paragraph."]}),
            }
        ),
        outputs=Ports(),
    )
