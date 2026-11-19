"""Tests for write_excel's layout options: how a tab looks, never what it holds.

A report written by a flow can match an older hand-written report figure for figure and format for
format, and still read worse. Its paragraphs run on as one long line. Its title is plain. Its
columns are all one width. Its data tabs have no filter dropdowns. These options fix that, one sheet
entry at a time, beside the ``formats`` rules each entry already carries:

- ``widths``: a stated width per column. A rule names a type or one column, like a format rule, and
  a column rule beats a type rule.
- ``fit_widths``: size every column to its contents, header included, up to a sensible cap. A
  stated width beats a fitted one, so a stated width is how a column gets past the cap.
- ``table``: make the tab's data a spreadsheet table, with filter dropdowns on the header. The name
  is stated, or derived from the tab name. The table has no colour style unless one of the
  spreadsheet's built-in styles is named.
- ``wrap``: wrap the text of the data cells a rule reaches, and align them to the top. The header
  is left alone.
- ``header``: make every heading bold and/or a stated size. On a notes tab the heading is the title.

The rulings pinned here:

**Layout never touches a figure.** Every option, alone or together, leaves every stored value and
every number format exactly as it would be without it. That is checked against the raw cell text in
the file, not only through a reader.

**No option, no layout.** A sheet entry that states none of these is written as it always was.

**An empty port still gets its tab, with its headers.** With a table asked for, the table covers
the header and one empty row. That is the smallest table a spreadsheet accepts, and it is what a
spreadsheet makes itself when a table is drawn over headers alone. Refusing a table on an empty tab
would stop a correct flow in a quiet month.

**Refused by name, before anything is written.** A rule about a column the frame does not carry, a
rule that cannot mean one thing, a table name a spreadsheet cannot use, two tables with one name
(names are compared ignoring case, as a spreadsheet compares them), a table name a template already
uses on a tab this step leaves alone, and a table over two headings that differ only by case.

Vocabulary is generic -- key, amount, note -- because the library carries no caller's nouns.
"""

import datetime as dt
import posixpath
import re
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import openpyxl
import polars as pl
import pytest
from openpyxl.styles import Font
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.table import Table

from dantron_etl.column_types import DECLARED_TYPES
from dantron_etl.authoring import ConfigError, DataError, Ports, check_tool, run_tool
from dantron_etl.config import expression_fields
from dantron_etl.tools.write_excel import WriteExcel

# ---------------------------------------------------------------------------
# Test data and helpers
# ---------------------------------------------------------------------------

MONEY_FORMAT = "#,##0.00"
COUNT_FORMAT = "0"
A_LONG_FIGURE = 661617.4665392789
"""More decimals than any format shows, so a value rewritten on the way in is visible."""

A_LARGE_FIGURE = 1234567.0
"""Shown under the money format as '1,234,567.00': twelve characters, longer than its own repr."""

A_LONG_NOTE = "a note long enough to need a wider column"

YAML_REWRITES = frozenset({"on", "off", "yes", "no", "true", "false", "null", "y", "n"})

_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_DOC_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

_TABLE_NAME = re.compile(r"^[A-Za-z_\\][A-Za-z0-9_.]*$")
_A1_LOOKALIKE = re.compile(r"^[A-Za-z]{1,3}[0-9]+$")
_R1C1_LOOKALIKE = re.compile(r"^[RrCc]$|^[Rr][0-9]*[Cc][0-9]*$|^[Rr][0-9]+$")


def _laid_out_rows(height: int = 3) -> pl.DataFrame:
    """Six columns: text, two money figures, a count, a date and a note with a gap in it.

    Columns A to F. The gaps are deliberate: a formatted cell with nothing in it is still a cell
    whose number format a layout option could disturb.
    """
    frame = pl.DataFrame(
        {
            "key": ["r1", "r2", "r3"],
            "amount": [A_LONG_FIGURE, None, A_LARGE_FIGURE],
            "balance": [0.125, 2.0, -99.999],
            "count": [1, 2, 3],
            "as_at": [dt.date(2026, 1, 31), dt.date(2026, 2, 28), dt.date(2026, 3, 31)],
            "note": ["short", A_LONG_NOTE, None],
        }
    )
    return frame.head(height)


_FORMATS: list[dict[str, Any]] = [
    {"type": "Float64", "format": MONEY_FORMAT},
    {"column": "count", "format": COUNT_FORMAT},
]

_ALL_OPTIONS: dict[str, Any] = {
    "widths": [{"type": "Float64", "width": 18}, {"column": "note", "width": 60}],
    "fit_widths": True,
    "wrap": [{"type": "Float64"}, {"column": "note"}],
    "table": {"name": "figures_detail", "style": "TableStyleLight1"},
    "header": {"bold": True, "size": 14},
}
"""Every layout option at once. Wrap reaches the money columns on purpose: an alignment written by
replacing a cell's whole style would take its number format with it."""


def _write(frames: Mapping[str, pl.DataFrame], config: Mapping[str, Any]) -> None:
    """Put ``frames`` through the boundary, the way a flow does."""
    result = run_tool(WriteExcel(), Ports(dict(frames)), dict(config))
    assert result.names == (), "a terminal sink produces no ports at all"


def _one_tab(target: Path, frame: pl.DataFrame, **options: Any) -> None:
    """Write ``frame`` to the tab 'Figures', with the money formats and the given layout options."""
    _write(
        {"one": frame},
        {"path": str(target), "sheets": [{"port": "one", "sheet": "Figures", "formats": _FORMATS, **options}]},
    )


def _sheet_xml(path: Path, sheet: str) -> ElementTree.Element:
    """The XML of one tab, found through the workbook's own relationships rather than by guessing."""
    with zipfile.ZipFile(path) as archive:
        book = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rid = next(s.get(f"{_DOC_REL}id") for s in book.iter(f"{_MAIN}sheet") if s.get("name") == sheet)
        target = next(rel.get("Target") for rel in rels if rel.get("Id") == rid)
        assert target is not None
        part = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))
        return ElementTree.fromstring(archive.read(part))


def _shared_strings(path: Path) -> list[str]:
    """The workbook's shared strings, in index order; empty if it has none."""
    with zipfile.ZipFile(path) as archive:
        if "xl/sharedStrings.xml" not in archive.namelist():
            return []
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(text.text or "" for text in item.iter(f"{_MAIN}t")) for item in root.iter(f"{_MAIN}si")]


def _stored(path: Path, sheet: str) -> dict[str, tuple[str | None, str]]:
    """Every cell holding something, as the file stores it: its type marker and its raw text.

    A shared string is looked up so two files that number their strings differently still compare.
    Everything else is the literal text inside ``<v>``: for a number, the exact digits stored.
    """
    shared = _shared_strings(path)
    stored: dict[str, tuple[str | None, str]] = {}
    for cell in _sheet_xml(path, sheet).iter(f"{_MAIN}c"):
        kind = cell.get("t")
        value = cell.find(f"{_MAIN}v")
        inline = cell.find(f"{_MAIN}is")
        if value is not None and value.text is not None:
            text = shared[int(value.text)] if kind == "s" else value.text
        elif inline is not None:
            text = "".join(t.text or "" for t in inline.iter(f"{_MAIN}t"))
        else:
            continue
        stored[str(cell.get("r"))] = (kind, text)
    return stored


def _cells(path: Path, sheet: str) -> dict[str, tuple[str, str, str]]:
    """Every cell holding a value or a number format: its value, its kind and its number format."""
    book = openpyxl.load_workbook(path)
    try:
        cells: dict[str, tuple[str, str, str]] = {}
        for row in book[sheet].iter_rows():
            for cell in row:
                if cell.value is None and cell.number_format == "General":
                    continue
                cells[cell.coordinate] = (repr(cell.value), cell.data_type, cell.number_format)
        return cells
    finally:
        book.close()


def _widths(path: Path, sheet: str) -> dict[int, float]:
    """The width each column carries, by column number from 1, read from the tab's own XML.

    Read from ``<cols>`` rather than through openpyxl, which hands back a default width for any
    column asked about, set or not. A column with no width set is missing from the result.
    """
    widths: dict[int, float] = {}
    for column in _sheet_xml(path, sheet).iter(f"{_MAIN}col"):
        width = column.get("width")
        if width is None:
            continue
        for index in range(int(column.get("min", "0")), int(column.get("max", "0")) + 1):
            widths[index] = float(width)
    return widths


def _tables(path: Path, sheet: str) -> list[dict[str, Any]]:
    """Each table on one tab: its name, its range, its filter range, its column names and its style."""
    book = openpyxl.load_workbook(path)
    try:
        found = []
        for table in book[sheet].tables.values():
            info = table.tableStyleInfo
            found.append(
                {
                    "name": table.displayName,
                    "ref": table.ref,
                    "filter": table.autoFilter.ref if table.autoFilter is not None else None,
                    "columns": [column.name for column in table.tableColumns],
                    "style": info.name if info is not None else None,
                }
            )
        return found
    finally:
        book.close()


def _sheet_filter(path: Path, sheet: str) -> str | None:
    """The tab's own filter range, outside any table. A tab with a table must not carry one too."""
    book = openpyxl.load_workbook(path)
    try:
        return book[sheet].auto_filter.ref
    finally:
        book.close()


def _fonts(path: Path, sheet: str) -> dict[str, tuple[bool, float | None]]:
    """Every written cell's font, as (bold, size)."""
    book = openpyxl.load_workbook(path)
    try:
        return {cell.coordinate: (bool(cell.font.b), cell.font.sz) for row in book[sheet].iter_rows() for cell in row}
    finally:
        book.close()


def _alignments(path: Path, sheet: str) -> dict[str, tuple[bool, str | None]]:
    """Every written cell's alignment, as (wraps, vertical)."""
    book = openpyxl.load_workbook(path)
    try:
        return {
            cell.coordinate: (bool(cell.alignment.wrap_text), cell.alignment.vertical)
            for row in book[sheet].iter_rows()
            for cell in row
        }
    finally:
        book.close()


def _refusal(error: BaseException) -> str:
    """The refusal's text, after checking it is about the option and not about an unknown field.

    Before these options exist every one of them is refused as an unknown field, and that message
    happens to contain words like 'header' and 'table'. A refusal test must not pass on it.
    """
    message = str(error)
    assert "unknown config field" not in message, f"refused as an unknown field, not for its value: {message}"
    return message


def _read_back(path: Path, sheet: str) -> pl.DataFrame:
    return pl.read_excel(path, sheet_name=sheet, drop_empty_cols=False, raise_if_empty=False)


def _fits(width: float, longest: int) -> bool:
    """Wide enough for the longest text, and not wildly wider.

    The upper bound allows padding, room for a filter button and a floor for very short columns,
    without letting a fit that ignored the contents pass.
    """
    return longest <= width <= max(longest * 1.25 + 5, 13)


def _is_usable_table_name(name: str) -> bool:
    """A name a spreadsheet accepts for a table: the right characters, and not a cell address."""
    return (
        len(name) <= 255
        and _TABLE_NAME.match(name) is not None
        and _A1_LOOKALIKE.match(name) is None
        and _R1C1_LOOKALIKE.match(name) is None
    )


def _template_with_tables(path: Path) -> Path:
    """A curated workbook: a hand-written Notes tab, and two tabs that each carry a table.

    'Notes' and 'Summary' are tabs the flow never names. 'Figures' is the tab it replaces. The
    tables are named 'Summary_table' and 'Figures'.
    """
    book = openpyxl.Workbook()
    book.remove(book.active)
    notes = book.create_sheet("Notes")
    notes["A1"] = "carried over: Notes"
    notes["A1"].font = Font(bold=True)
    notes["B2"] = 42
    notes["C3"] = "=SUM(B2:B2)"
    notes.column_dimensions["A"].width = 30
    for tab, table_name in (("Summary", "Summary_table"), ("Figures", "Figures")):
        sheet = book.create_sheet(tab)
        sheet.append(["label", "total"])
        sheet.append(["x", 1])
        sheet.append(["y", 2])
        sheet.add_table(Table(displayName=table_name, ref="A1:B3"))
    book.defined_names["carried_total"] = DefinedName("carried_total", attr_text="'Notes'!$B$2")
    book.save(path)
    book.close()
    return path


# ---------------------------------------------------------------------------
# Each option lays out exactly what it states
# ---------------------------------------------------------------------------


class TestEachOptionLaysOutWhatItStates:
    """Read the written file back and find exactly the stated layout, and nothing extra."""

    def test_a_width_rule_sets_its_width_on_every_column_it_reaches_and_no_other(self, tmp_path: Path) -> None:
        """A type rule reaches both money columns. A column rule reaches one column. The rest keep no width."""
        target = tmp_path / "pack.xlsx"

        _one_tab(
            target,
            _laid_out_rows(),
            widths=[{"type": "Float64", "width": 18}, {"column": "note", "width": 60}],
        )

        assert _widths(target, "Figures") == {2: 18, 3: 18, 6: 60}

    @pytest.mark.parametrize(
        "rules",
        [
            pytest.param(
                [{"type": "Float64", "width": 18}, {"column": "amount", "width": 25}],
                id="the_general_rule_written_first",
            ),
            pytest.param(
                [{"column": "amount", "width": 25}, {"type": "Float64", "width": 18}],
                id="the_exception_written_first",
            ),
        ],
    )
    def test_a_column_width_beats_the_type_width_whichever_order_they_were_written_in(
        self,
        tmp_path: Path,
        rules: list[dict[str, Any]],
    ) -> None:
        """The same precedence as format rules: the kind of rule decides, not where it sits in the list."""
        target = tmp_path / "pack.xlsx"

        _one_tab(target, _laid_out_rows(), widths=rules)

        assert _widths(target, "Figures") == {2: 25, 3: 18}

    def test_fitting_sizes_every_column_to_its_longest_text_header_included(self, tmp_path: Path) -> None:
        """Each column is at least as wide as its header and its longest value, and not much wider.

        The money column is the one that matters. '1,234,567.00' is twelve characters on screen and
        its stored value is shorter. A column fitted to the stored digits shows '####'.
        """
        target = tmp_path / "pack.xlsx"

        _one_tab(target, _laid_out_rows(), fit_widths=True)

        widths = _widths(target, "Figures")
        assert set(widths) == {1, 2, 3, 4, 5, 6}, "fitting sizes every column on the tab"
        for index, header in enumerate(_laid_out_rows().columns, start=1):
            assert widths[index] >= len(header), f"{header} is narrower than its own heading"
        assert _fits(widths[1], len("key")), f"a short text column was not fitted: {widths[1]}"
        assert _fits(widths[6], len(A_LONG_NOTE)), f"the note column was not fitted to its longest note: {widths[6]}"
        assert widths[2] >= len("1,234,567.00"), "a fitted money column is too narrow for its figure as shown"
        assert widths[5] >= len("2026-01-31"), "a fitted date column is too narrow for its date"

    def test_fitting_caps_a_column_of_paragraphs_at_a_sensible_width(self, tmp_path: Path) -> None:
        """A thousand-character paragraph does not make a thousand-character column."""
        target = tmp_path / "pack.xlsx"
        frame = pl.DataFrame({"key": ["r1", "r2"], "paragraph": ["word " * 200, "short"]})

        _write(
            {"one": frame},
            {"path": str(target), "sheets": [{"port": "one", "sheet": "Notes", "fit_widths": True}]},
        )

        widths = _widths(target, "Notes")
        assert 30 <= widths[2] <= 100, f"the paragraph column was not capped sensibly: {widths[2]}"
        assert _fits(widths[1], len("key")), "the cap reached a column that did not need it"

    def test_a_stated_width_beats_fitting_and_is_how_a_column_gets_past_the_cap(self, tmp_path: Path) -> None:
        target = tmp_path / "pack.xlsx"
        frame = pl.DataFrame({"key": ["r1", "r2"], "paragraph": ["word " * 200, "short"]})

        _write(
            {"one": frame},
            {
                "path": str(target),
                "sheets": [
                    {
                        "port": "one",
                        "sheet": "Notes",
                        "fit_widths": True,
                        "widths": [{"column": "paragraph", "width": 150}],
                    }
                ],
            },
        )

        widths = _widths(target, "Notes")
        assert widths[2] == 150, "a stated width lost to the fit"
        assert _fits(widths[1], len("key")), "the column nobody stated a width for was not fitted"

    @pytest.mark.parametrize(
        ("table", "expected_style"),
        [
            pytest.param({"name": "figures_detail"}, None, id="no_style_stated"),
            pytest.param({"name": "figures_detail", "style": "TableStyleLight1"}, "TableStyleLight1", id="a_style"),
        ],
    )
    def test_a_table_covers_the_header_and_every_row_with_filter_dropdowns(
        self,
        tmp_path: Path,
        table: dict[str, Any],
        expected_style: str | None,
    ) -> None:
        """One table, the stated name, the full range, a filter over it, and its columns named as the headers.

        Column names that differ from the header cells make a spreadsheet repair the file on open. A
        filter on the tab as well as on the table does the same. No style is stated, so none is
        written: a plain table with dropdowns, as the older report had.
        """
        target = tmp_path / "pack.xlsx"

        _one_tab(target, _laid_out_rows(), table=table)

        assert _tables(target, "Figures") == [
            {
                "name": "figures_detail",
                "ref": "A1:F4",
                "filter": "A1:F4",
                "columns": ["key", "amount", "balance", "count", "as_at", "note"],
                "style": expected_style,
            }
        ]
        assert _sheet_filter(target, "Figures") is None, "a tab filter beside a table's filter breaks the file"

    @pytest.mark.parametrize(
        ("tab", "expected"),
        [
            pytest.param("Figures", "Figures", id="a_plain_tab_name"),
            pytest.param("Monthly Detail", "Monthly_Detail", id="a_space_becomes_an_underscore"),
            pytest.param("Net-Total", "Net_Total", id="a_hyphen_becomes_an_underscore"),
            pytest.param("2026 Detail", None, id="a_tab_name_starting_with_a_digit"),
            pytest.param("Q3", None, id="a_tab_name_that_is_a_cell_address"),
            pytest.param("Net - Total (all)", None, id="punctuation"),
        ],
    )
    def test_a_table_name_left_out_is_derived_from_the_tab_name(
        self,
        tmp_path: Path,
        tab: str,
        expected: str | None,
    ) -> None:
        """A plain tab name is used as it is. Any other becomes a name a spreadsheet accepts.

        A character a table name cannot carry becomes an underscore. A name that would start with a
        digit, or read as a cell address, is changed until it does neither. Only the result's
        validity is pinned for those, not its exact spelling.
        """
        target = tmp_path / "pack.xlsx"

        _write(
            {"one": _laid_out_rows()},
            {"path": str(target), "sheets": [{"port": "one", "sheet": tab, "table": {}}]},
        )

        tables = _tables(target, tab)
        assert len(tables) == 1 and tables[0]["ref"] == "A1:F4"
        name = tables[0]["name"]
        assert _is_usable_table_name(name), f"the derived table name {name!r} is not one a spreadsheet accepts"
        if expected is not None:
            assert name == expected

    @pytest.mark.parametrize(
        ("rule", "wrapped"),
        [
            pytest.param({"column": "note"}, {"F"}, id="a_column_rule"),
            pytest.param({"type": "String"}, {"A", "F"}, id="a_type_rule"),
        ],
    )
    def test_wrap_wraps_and_top_aligns_every_data_cell_it_reaches_and_no_other_cell(
        self,
        tmp_path: Path,
        rule: dict[str, Any],
        wrapped: set[str],
    ) -> None:
        """Every data cell of a reached column, the empty one included. The header is left alone."""
        target = tmp_path / "pack.xlsx"

        _one_tab(target, _laid_out_rows(), wrap=[rule])

        for coordinate, alignment in _alignments(target, "Figures").items():
            column, row = coordinate[0], int(coordinate[1:])
            if row > 1 and column in wrapped:
                assert alignment == (True, "top"), f"{coordinate} should wrap and sit at the top"
            else:
                assert alignment == (False, None), f"{coordinate} was aligned by a rule that does not reach it"

    @pytest.mark.parametrize(
        ("header", "bold", "size"),
        [
            pytest.param({"bold": True, "size": 14}, True, 14, id="bold_and_a_size"),
            pytest.param({"bold": True}, True, None, id="bold_only"),
            pytest.param({"size": 14}, False, 14, id="a_size_only"),
        ],
    )
    def test_the_header_style_reaches_every_heading_and_no_data_cell(
        self,
        tmp_path: Path,
        header: dict[str, Any],
        bold: bool,
        size: float | None,
    ) -> None:
        """What is stated changes. What is not stated keeps the workbook's own default."""
        target = tmp_path / "pack.xlsx"

        _one_tab(target, _laid_out_rows(), header=header)

        fonts = _fonts(target, "Figures")
        default_size = fonts["A2"][1]
        for coordinate, (is_bold, cell_size) in fonts.items():
            if coordinate[1:] == "1":
                assert is_bold is bold, f"heading {coordinate} bold should be {bold}"
                assert cell_size == (size if size is not None else default_size), f"heading {coordinate} size"
            else:
                assert is_bold is False, f"data cell {coordinate} was made bold"
                assert cell_size == default_size, f"data cell {coordinate} changed size"

    def test_a_notes_tab_reads_as_a_title_and_wrapped_paragraphs(self, tmp_path: Path) -> None:
        """The case these options were asked for: a one-column notes tab whose heading is its title.

        A wide column, a bold 14-point title, and paragraphs that wrap. The blank line between the
        paragraphs stays blank.
        """
        target = tmp_path / "pack.xlsx"
        title = "Report notes - as at 2026-01-31"
        frame = pl.DataFrame({title: ["The first paragraph. " * 20, None, "The second paragraph. " * 20]})

        _write(
            {"notes": frame},
            {
                "path": str(target),
                "sheets": [
                    {
                        "port": "notes",
                        "sheet": "Read Me",
                        "widths": [{"type": "String", "width": 120}],
                        "wrap": [{"type": "String"}],
                        "header": {"bold": True, "size": 14},
                    }
                ],
            },
        )

        assert _widths(target, "Read Me") == {1: 120}
        fonts = _fonts(target, "Read Me")
        assert fonts["A1"] == (True, 14), "the title is not bold at 14 points"
        alignments = _alignments(target, "Read Me")
        assert alignments["A1"] == (False, None), "the title was wrapped"
        for coordinate in ("A2", "A3", "A4"):
            assert alignments[coordinate] == (True, "top"), f"paragraph cell {coordinate} does not wrap"
        assert _read_back(target, "Read Me").columns == [title]
        assert _stored(target, "Read Me")["A2"][1] == "The first paragraph. " * 20


# ---------------------------------------------------------------------------
# Layout never touches a figure
# ---------------------------------------------------------------------------


class TestLayoutNeverTouchesAFigure:
    """The property that matters most. Layout changes how a tab looks, never a value or its format."""

    @pytest.mark.parametrize("height", [3, 0], ids=["rows_arrived", "no_rows_arrived"])
    @pytest.mark.parametrize(
        "options",
        [
            *(pytest.param({name: value}, id=name) for name, value in _ALL_OPTIONS.items()),
            pytest.param(_ALL_OPTIONS, id="every_option_together"),
        ],
    )
    def test_every_stored_value_and_number_format_is_what_it_is_without_layout(
        self,
        tmp_path: Path,
        options: dict[str, Any],
        height: int,
    ) -> None:
        """The same frame written twice, once plain and once laid out, compared cell by cell.

        Three comparisons. The raw text the file stores for every cell, which is the exact digits of
        a number. Every cell's value, kind and number format as openpyxl reads them, which includes
        the formatted cells with nothing in them. And the frame a reader gets back.
        """
        plain = tmp_path / "plain.xlsx"
        laid_out = tmp_path / "laid_out.xlsx"

        _one_tab(plain, _laid_out_rows(height))
        _one_tab(laid_out, _laid_out_rows(height), **options)

        assert _stored(laid_out, "Figures") == _stored(plain, "Figures"), "a layout option changed a stored value"
        assert _cells(laid_out, "Figures") == _cells(plain, "Figures"), "a layout option changed a value or format"
        assert _read_back(laid_out, "Figures").equals(_read_back(plain, "Figures"))
        if height:
            stored_figure = float(_stored(laid_out, "Figures")["B2"][1])
            assert stored_figure == A_LONG_FIGURE, f"the long figure was stored as {stored_figure!r}"

    @pytest.mark.parametrize(
        "options",
        [
            pytest.param({}, id="nothing_stated"),
            pytest.param({"fit_widths": False}, id="fitting_turned_off"),
            pytest.param({"widths": [], "wrap": []}, id="empty_lists_of_rules"),
        ],
    )
    def test_with_no_layout_option_a_tab_has_no_layout(self, tmp_path: Path, options: dict[str, Any]) -> None:
        """No width, no table, no filter, no bold heading, no wrapped cell: the tab as it was written before."""
        target = tmp_path / "pack.xlsx"

        _one_tab(target, _laid_out_rows(), **options)

        assert _widths(target, "Figures") == {}, "a column was given a width nobody asked for"
        assert _tables(target, "Figures") == [], "a table nobody asked for"
        assert _sheet_filter(target, "Figures") is None
        assert not any(bold for bold, _ in _fonts(target, "Figures").values()), "a cell was made bold"
        assert not any(wraps for wraps, _ in _alignments(target, "Figures").values()), "a cell was wrapped"

    def test_one_tabs_layout_does_not_reach_the_tab_beside_it(self, tmp_path: Path) -> None:
        target = tmp_path / "pack.xlsx"

        _write(
            {"one": _laid_out_rows(), "two": _laid_out_rows()},
            {
                "path": str(target),
                "sheets": [
                    {"port": "one", "sheet": "Figures", **_ALL_OPTIONS},
                    {"port": "two", "sheet": "Plain"},
                ],
            },
        )

        assert len(_tables(target, "Figures")) == 1, "the laid-out tab lost its table"
        assert _widths(target, "Plain") == {}
        assert _tables(target, "Plain") == []
        assert not any(bold for bold, _ in _fonts(target, "Plain").values())
        assert not any(wraps for wraps, _ in _alignments(target, "Plain").values())


# ---------------------------------------------------------------------------
# An empty port, with layout
# ---------------------------------------------------------------------------


class TestAnEmptyPortWithLayout:
    """The owner's ruling still holds: an empty port gets its tab and its headers, options and all."""

    def test_an_empty_port_gets_its_headers_and_every_option_it_asked_for(self, tmp_path: Path) -> None:
        """The table covers the header and one empty row: the smallest table a spreadsheet accepts.

        The tab still reads as 'nothing this month'. The empty row holds no value, so a reader gets
        no rows back.
        """
        target = tmp_path / "pack.xlsx"

        _one_tab(target, _laid_out_rows(0), **_ALL_OPTIONS)

        back = _read_back(target, "Figures")
        assert back.columns == ["key", "amount", "balance", "count", "as_at", "note"]
        assert back.height == 0, "an empty port came back with rows"
        assert set(_stored(target, "Figures")) == {"A1", "B1", "C1", "D1", "E1", "F1"}, "a value under the header"
        assert _tables(target, "Figures") == [
            {
                "name": "figures_detail",
                "ref": "A1:F2",
                "filter": "A1:F2",
                "columns": ["key", "amount", "balance", "count", "as_at", "note"],
                "style": "TableStyleLight1",
            }
        ]
        widths = _widths(target, "Figures")
        assert widths[2] == 18 and widths[3] == 18 and widths[6] == 60, "a stated width was lost on an empty tab"
        for index, header in ((1, "key"), (4, "count"), (5, "as_at")):
            assert widths[index] >= len(header), f"{header} was fitted narrower than its heading on an empty tab"
        assert _fonts(target, "Figures")["A1"] == (True, 14)


# ---------------------------------------------------------------------------
# Layout inside a template
# ---------------------------------------------------------------------------


class TestLayoutInATemplate:
    """Layout on a tab written into a template, and the tabs the step does not name left alone."""

    @pytest.mark.parametrize(
        ("table", "expected"),
        [
            pytest.param({"name": "Figures"}, ["Figures"], id="the_replaced_tabs_table_name_reused"),
            pytest.param(None, [], id="no_table_asked_for"),
        ],
    )
    def test_layout_on_a_replaced_tab_leaves_every_other_tab_as_it_was(
        self,
        tmp_path: Path,
        table: dict[str, Any] | None,
        expected: list[str],
    ) -> None:
        """The replaced tab's old table goes with it, so its name is free to use again.

        Asking for no table leaves the replaced tab with none. The Summary tab's table, the Notes
        tab's bold, formula and width, and the workbook's defined name all come through.
        """
        template = _template_with_tables(tmp_path / "template.xlsx")
        target = tmp_path / "pack.xlsx"
        entry: dict[str, Any] = {
            "port": "one",
            "sheet": "Figures",
            "widths": [{"column": "note", "width": 60}],
            "header": {"bold": True},
        }
        if table is not None:
            entry["table"] = table

        _write(
            {"one": _laid_out_rows()},
            {"path": str(target), "template": str(template), "overwrite_sheets": True, "sheets": [entry]},
        )

        assert [t["name"] for t in _tables(target, "Figures")] == expected
        if expected:
            assert _tables(target, "Figures")[0]["ref"] == "A1:F4"
        assert _widths(target, "Figures") == {6: 60}
        assert _fonts(target, "Figures")["A1"][0] is True
        assert [t["name"] for t in _tables(target, "Summary")] == ["Summary_table"], "an untouched tab lost its table"
        assert _widths(target, "Notes") == {1: 30}, "an untouched tab lost its column width"
        book = openpyxl.load_workbook(target)
        try:
            notes = book["Notes"]
            assert notes["A1"].value == "carried over: Notes" and notes["A1"].font.b is True
            assert notes["C3"].value == "=SUM(B2:B2)"
            assert "carried_total" in book.defined_names
        finally:
            book.close()

    @pytest.mark.parametrize("name", ["Summary_table", "summary_TABLE"], ids=["same_spelling", "other_case"])
    def test_a_table_name_already_used_on_a_tab_left_alone_is_refused_before_anything_is_written(
        self,
        tmp_path: Path,
        name: str,
    ) -> None:
        """Two tables of one name break the file. The name is still in use, so the new one is refused.

        This can only be known with the template open. It is still refused before the save, so no
        pack is left behind and the template is untouched.
        """
        template = _template_with_tables(tmp_path / "template.xlsx")
        before = template.read_bytes()
        target = tmp_path / "pack.xlsx"

        with pytest.raises(ConfigError) as caught:
            _write(
                {"one": _laid_out_rows()},
                {
                    "path": str(target),
                    "template": str(template),
                    "sheets": [{"port": "one", "sheet": "New tab", "table": {"name": name}}],
                },
            )

        message = _refusal(caught.value)
        assert name in message, f"the refusal does not name the table: {message}"
        assert "Summary" in message, f"the refusal does not name the tab already using it: {message}"
        assert not target.exists(), "a refused write left a pack behind"
        assert template.read_bytes() == before, "a refused write touched the template"


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class TestRefusalsAboutLayout:
    """Each refusal names what to change, and comes before anything is written, rows or no rows."""

    @pytest.mark.parametrize("height", [3, 0], ids=["rows_arrived", "no_rows_arrived"])
    @pytest.mark.parametrize(
        ("options", "field"),
        [
            pytest.param({"widths": [{"column": "notes", "width": 40}]}, "widths", id="a_width"),
            pytest.param({"wrap": [{"column": "notes"}]}, "wrap", id="a_wrap"),
        ],
    )
    def test_a_rule_naming_a_column_the_frame_does_not_carry_is_refused_by_name(
        self,
        tmp_path: Path,
        options: dict[str, Any],
        field: str,
        height: int,
    ) -> None:
        """A typo for 'note'. Passing over it would deliver a tab laid out differently and say nothing."""
        target = tmp_path / "pack.xlsx"

        with pytest.raises((ConfigError, DataError)) as caught:
            _one_tab(target, _laid_out_rows(height), **options)

        message = _refusal(caught.value)
        assert "'notes'" in message, f"the refusal does not name the missing column: {message}"
        assert field in message, f"the refusal does not name the field: {message}"
        assert not target.exists(), "a refused write left a pack behind"

    @pytest.mark.parametrize("height", [3, 0], ids=["rows_arrived", "no_rows_arrived"])
    @pytest.mark.parametrize(
        ("options", "named"),
        [
            pytest.param(
                {"widths": [{"type": "Float64", "column": "amount", "width": 18}]},
                ("type", "column"),
                id="a_width_rule_with_both_a_type_and_a_column",
            ),
            pytest.param({"widths": [{"width": 18}]}, ("type", "column"), id="a_width_rule_with_neither"),
            pytest.param(
                {"widths": [{"type": "Float64", "width": 18}, {"type": "Float64", "width": 25}]},
                ("Float64",),
                id="a_type_given_two_widths",
            ),
            pytest.param(
                {"widths": [{"column": "amount", "width": 18}, {"column": "amount", "width": 25}]},
                ("amount",),
                id="a_column_given_two_widths",
            ),
            pytest.param({"widths": [{"column": "amount", "width": 0}]}, ("width",), id="a_width_of_zero"),
            pytest.param({"widths": [{"column": "amount", "width": -5}]}, ("width", "-5"), id="a_negative_width"),
            pytest.param({"widths": [{"column": "amount", "width": 256}]}, ("width", "256"), id="a_width_too_wide"),
            pytest.param(
                {"wrap": [{"type": "String", "column": "note"}]}, ("type", "column"), id="a_wrap_rule_with_both"
            ),
            pytest.param({"wrap": [{}]}, ("type", "column"), id="a_wrap_rule_with_neither"),
            pytest.param({"wrap": [{"column": "note"}, {"column": "note"}]}, ("note",), id="a_column_wrapped_twice"),
            pytest.param({"header": {}}, ("header",), id="a_header_style_stating_nothing"),
            pytest.param({"header": {"size": 0}}, ("size",), id="a_heading_size_of_zero"),
            pytest.param({"header": {"size": 410}}, ("size", "410"), id="a_heading_size_too_large"),
            pytest.param({"table": {"style": "Fancy"}}, ("style", "Fancy"), id="a_table_style_that_is_not_built_in"),
            pytest.param({"table": {"name": "my table"}}, ("name", "my table"), id="a_table_name_with_a_space"),
            pytest.param({"table": {"name": "1st"}}, ("name", "1st"), id="a_table_name_starting_with_a_digit"),
            pytest.param({"table": {"name": "A1"}}, ("name", "A1"), id="a_table_name_that_is_a_cell_address"),
            pytest.param({"table": {"name": "R1C1"}}, ("name", "R1C1"), id="a_table_name_that_is_an_r1c1_address"),
            pytest.param({"table": {"name": "t" * 256}}, ("name",), id="a_table_name_too_long"),
        ],
    )
    def test_an_option_that_cannot_mean_one_thing_is_refused_with_the_field_named(
        self,
        tmp_path: Path,
        options: dict[str, Any],
        named: tuple[str, ...],
        height: int,
    ) -> None:
        target = tmp_path / "pack.xlsx"

        with pytest.raises(ConfigError) as caught:
            _one_tab(target, _laid_out_rows(height), **options)

        message = _refusal(caught.value)
        for word in named:
            assert word in message, f"the refusal does not name {word!r}: {message}"
        assert not target.exists(), "a refused write left a pack behind"

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            pytest.param(("Figures", {"name": "detail"}), ("Extra", {"name": "detail"}), id="two_stated_alike"),
            pytest.param(("Figures", {"name": "detail"}), ("Extra", {"name": "DETAIL"}), id="alike_but_for_case"),
            pytest.param(("Figures", {}), ("Extra", {"name": "figures"}), id="a_derived_name_and_a_stated_one"),
            pytest.param(("Net-Total", {}), ("Net Total", {}), id="two_derived_names"),
        ],
    )
    def test_two_tables_of_one_name_are_refused_with_both_tabs_named(
        self,
        tmp_path: Path,
        first: tuple[str, dict[str, Any]],
        second: tuple[str, dict[str, Any]],
    ) -> None:
        """A spreadsheet compares table names ignoring case, so this does too."""
        target = tmp_path / "pack.xlsx"

        with pytest.raises(ConfigError) as caught:
            _write(
                {"one": _laid_out_rows(), "two": _laid_out_rows()},
                {
                    "path": str(target),
                    "sheets": [
                        {"port": "one", "sheet": first[0], "table": first[1]},
                        {"port": "two", "sheet": second[0], "table": second[1]},
                    ],
                },
            )

        message = _refusal(caught.value)
        for tab in (first[0], second[0]):
            assert f"'{tab}'" in message, f"the refusal does not name the tab {tab!r}: {message}"
        assert "table" in message
        assert not target.exists(), "a refused write left a pack behind"

    @pytest.mark.parametrize("height", [3, 0], ids=["rows_arrived", "no_rows_arrived"])
    def test_a_table_over_two_headings_that_differ_only_by_case_is_refused(self, tmp_path: Path, height: int) -> None:
        """A table's column names must differ ignoring case. A frame's need not. The table is refused."""
        target = tmp_path / "pack.xlsx"
        frame = pl.DataFrame({"Amount": [1.0, 2.0, 3.0], "amount": [4.0, 5.0, 6.0]}).head(height)

        with pytest.raises((ConfigError, DataError)) as caught:
            _write(
                {"one": frame},
                {"path": str(target), "sheets": [{"port": "one", "sheet": "Figures", "table": {}}]},
            )

        message = _refusal(caught.value)
        assert "'Amount'" in message and "'amount'" in message, f"the refusal does not name both: {message}"
        assert not target.exists(), "a refused write left a pack behind"


# ---------------------------------------------------------------------------
# Declared and documented
# ---------------------------------------------------------------------------


class TestLayoutIsDeclaredAndDocumented:
    """The options as a definition file meets them: optional, typed, spelled safely, and demonstrated."""

    def test_the_sheet_entry_declares_each_option_with_its_shape(self) -> None:
        entry = WriteExcel().config_template()["sheets"].items
        assert entry is not None and entry.fields is not None
        fields = entry.fields

        assert set(fields) == {"port", "sheet", "formats", "widths", "fit_widths", "wrap", "table", "header"}
        for name in ("widths", "fit_widths", "wrap", "table", "header"):
            assert fields[name].required is False, f"{name} is optional: a tab that states nothing is written as before"

        assert fields["fit_widths"].type == "boolean"

        widths = fields["widths"]
        assert widths.type == "array" and widths.items is not None and widths.items.fields is not None
        assert set(widths.items.fields) == {"type", "column", "width"}
        assert widths.items.fields["width"].type == "number" and widths.items.fields["width"].required is True
        assert tuple(widths.items.fields["type"].enum or ()) == tuple(DECLARED_TYPES)

        wrap = fields["wrap"]
        assert wrap.type == "array" and wrap.items is not None and wrap.items.fields is not None
        assert set(wrap.items.fields) == {"type", "column"}
        assert tuple(wrap.items.fields["type"].enum or ()) == tuple(DECLARED_TYPES)

        table = fields["table"]
        assert table.type == "object" and table.fields is not None
        assert set(table.fields) == {"name", "style"}
        assert table.fields["name"].required is False, "a name left out is derived from the tab"
        assert table.fields["style"].required is False, "no style stated is a plain table"

        header = fields["header"]
        assert header.type == "object" and header.fields is not None
        assert set(header.fields) == {"bold", "size"}
        assert header.fields["bold"].type == "boolean"
        assert header.fields["size"].type == "number"

        every_name = {"widths", "fit_widths", "wrap", "table", "header", "width", "name", "style", "bold", "size"}
        assert not every_name & YAML_REWRITES, "a field a definition file rewrites is not a field"
        assert expression_fields(WriteExcel().config_template()) == (), "no layout option is a column reference"

    def test_a_worked_example_uses_every_option_and_really_runs(self) -> None:
        """The catalog shows each option in use, and the conformance check runs that example."""
        tool = WriteExcel()
        help_ = tool.help()

        used = {key for example in help_.examples for entry in example.configuration["sheets"] for key in entry}
        missing = {"widths", "fit_widths", "wrap", "table", "header"} - used
        assert not missing, f"no worked example shows: {sorted(missing)}"

        documented = " ".join((help_.detailed_description, *help_.notes, *help_.field_help.values())).lower()
        for word in ("table", "wrap", "width"):
            assert word in documented, f"the help never mentions {word!r}"

        check_tool(tool)
