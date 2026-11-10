"""Tests for the ReadExcel tool: one named tab of one XLSX workbook becomes one frame.

ReadExcel declares **no input ports** -- the contract constrains only ``output_ports``, so a source
is expressible -- and one output port, ``rows``. It handles XLSX and nothing else: a spreadsheet has
sheets and typed cells, a plain text file has neither, and the hazards are not the same hazards. A
delimited-text source is therefore a different tool, and this one's refusals name the format so a
reader who arrived at the wrong tool is told which one they are holding.

The rulings pinned here, each argued rather than assumed:

**The tab is named, and naming it is not optional.** A workbook with one tab and a workbook with
twelve are the same file type, so "the first tab" is a positional read wearing a default's clothes:
insert a cover sheet in front of the data -- which is exactly how a delivered pack is laid out, a
Read Me tab, then a Checks tab, then the figures -- and the flow silently reads the cover sheet and
produces a plausible frame. A tab named in the configuration costs one line and turns that failure
into a refusal. An index is offered nowhere at all, for the same reason: it survives somebody
reordering the tabs, and it survives it silently.

**A tab that is not there is refused with the tabs that are listed beside it.** The two mistakes
made here are a misspelling and a wrong case, and both are obvious the instant the real list sits
next to them -- the same reasoning the library already applies to a column that is not on a port. A
name differing only in case or in a trailing space is one of those mistakes, not a match.

**A header row that repeats or omits a name is refused, not repaired.** A second column headed
``key`` arrives renamed ``key_1``, and a blank heading arrives as a positional placeholder; in both
cases a configuration downstream that names ``key`` gets whichever one came first, and nothing tells
anybody. A sheet's header row is its schema, and a schema that had to be guessed is refused.

**What comes back is what the sheet holds, not what somebody once put in.** A column of whole
numbers written as floats comes back as whole numbers; a column mixing text and figures comes back
as text. The reader states this rather than trying to restore an original dtype it cannot know,
because a reader that guessed would be right most of the time and wrong in a way no run reports.

**A refusal about a file is a refusal an author can act on.** A missing file, a missing tab, a path
that is not a workbook and a tab with nothing in it are all reported by this library, in its own
words, naming the path -- never as the exception the reader underneath happens to raise, which names
neither the flow nor anything the author wrote and which volunteers advice about the reader's own
keyword arguments.

Vocabulary is deliberately generic -- key, amount, code -- because the library carries no caller's
nouns.
"""

import datetime as dt
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import xlsxwriter
from polars.testing import assert_frame_equal

from dantron_etl.tool import (
    CATEGORIES,
    REGISTRY,
    ConfigError,
    ExecutionError,
    Ports,
    check_registry,
    check_tool,
    expression_fields,
    run_tool,
)
from dantron_etl.tools.read_excel import ReadExcel
from dantron_etl.tools.write_excel import WriteExcel

# ---------------------------------------------------------------------------
# Fixtures -- workbooks of the shape a delivered pack actually has
# ---------------------------------------------------------------------------

Sheets = Mapping[str, Sequence[Sequence[Any]]]


def _workbook(path: Path, sheets: Sheets) -> Path:
    """Write ``sheets`` cell by cell, so a test can lay out a sheet the way a person would.

    Deliberately hand-built rather than produced by the paired writer: several of the layouts below
    -- a title block above the table, a repeated heading, a tab with nothing on it -- are exactly
    the layouts the writer refuses to produce, and they are the ones a workbook that arrived from
    somebody else has.
    """
    with xlsxwriter.Workbook(str(path)) as workbook:
        date_format = workbook.add_format({"num_format": "yyyy-mm-dd"})
        for name, rows in sheets.items():
            sheet = workbook.add_worksheet(name)
            for row_index, row in enumerate(rows):
                for column_index, value in enumerate(row):
                    if isinstance(value, (dt.date, dt.datetime)):
                        sheet.write_datetime(row_index, column_index, value, date_format)
                    elif value is None:
                        sheet.write_blank(row_index, column_index, None)
                    else:
                        sheet.write(row_index, column_index, value)
    return path


def _pack(path: Path) -> Path:
    """A three-tab pack: a cover sheet, a checks tab, and the figures -- in that order.

    The order is the point. The figures are the last tab, so any reader that quietly falls back to
    the first one produces a frame that looks like data and is a cover sheet.
    """
    return _workbook(
        path,
        {
            "Read Me": [["This pack covers the period to 31 January 2026."]],
            "Checks": [["check", "outcome"], ["totals agree", "yes"]],
            "Figures": [
                ["key", "amount", "code"],
                ["r1", 10.25, "A"],
                ["r2", -5.5, "B"],
                ["r3", 1234.56, "A"],
            ],
        },
    )


def _read(config: Mapping[str, Any]) -> pl.DataFrame:
    """``config`` put through the boundary with no input ports at all, and ``rows`` unwrapped."""
    result = run_tool(ReadExcel(), Ports(), dict(config))
    assert result.names == ("rows",), "a source has one output port and no inputs"
    return result["rows"]


# ---------------------------------------------------------------------------
# Which tab, and what comes out of it
# ---------------------------------------------------------------------------


class TestWhichTabIsRead:
    """The tab named in the configuration, and no other."""

    @pytest.mark.parametrize(
        ("sheet", "columns", "first_row"),
        [
            pytest.param("Figures", ["key", "amount", "code"], ["r1", 10.25, "A"], id="the_data_tab"),
            pytest.param("Checks", ["check", "outcome"], ["totals agree", "yes"], id="the_checks_tab"),
        ],
    )
    def test_the_tab_named_in_the_configuration_is_the_tab_that_is_read(
        self,
        tmp_path: Path,
        sheet: str,
        columns: list[str],
        first_row: list[Any],
    ) -> None:
        """Pinned against a real multi-tab pack, because a single-sheet fixture proves nothing.

        A reader that ignored its configuration, or that fell back to the first tab, passes every
        test written against a workbook with one tab in it. Against this fixture it hands back a
        cover sheet instead, which is the failure this whole library keeps refusing: a plausible
        frame nobody questions.

        Column order is the sheet's own, left to right, because a spreadsheet is read by a person
        and the order they laid the columns out in is information.
        """
        rows = _read({"path": str(_pack(tmp_path / "pack.xlsx")), "sheet": sheet})

        assert rows.columns == columns
        assert rows.row(0) == tuple(first_row)

    @pytest.mark.parametrize(
        ("cells", "expected_dtype", "expected_values"),
        [
            pytest.param(
                [[dt.date(2026, 1, 31)], [dt.date(2026, 2, 28)]],
                pl.Date,
                [dt.date(2026, 1, 31), dt.date(2026, 2, 28)],
                id="a_date_cell_is_a_date_and_not_the_number_of_days_behind_it",
            ),
            pytest.param(
                [[10.25], [-5.5]],
                pl.Float64,
                [10.25, -5.5],
                id="a_money_figure_is_a_figure",
            ),
            pytest.param(
                [[10.0], [20.0]],
                pl.Int64,
                [10, 20],
                id="but_a_figure_with_nothing_after_the_point_comes_back_whole",
            ),
            pytest.param(
                [[10.25], ["not a number"]],
                pl.String,
                ["10.25", "not a number"],
                id="and_a_column_holding_both_comes_back_as_text",
            ),
            pytest.param(
                [["seen"], [None]],
                pl.String,
                ["seen", None],
                id="a_blank_cell_is_nothing_rather_than_an_empty_string",
            ),
        ],
    )
    def test_a_cell_arrives_as_whatever_the_sheet_actually_holds(
        self,
        tmp_path: Path,
        cells: list[list[Any]],
        expected_dtype: pl.DataType,
        expected_values: list[Any],
    ) -> None:
        """The reader reports the sheet; it does not reconstruct what was in the frame before it.

        The third and fourth cases are the two surprises, and they are pinned rather than fixed. A
        column of round figures comes back whole because a cell has no way to say "this was a
        decimal", and a column with one piece of text in it comes back entirely as text -- so the
        figures in it arrive as ``"10.25"``, and any downstream arithmetic on them fails at the
        arithmetic rather than here. A reader that guessed a dtype back would be right most months
        and wrong in a way no run reports; a flow that needs a particular dtype casts, in writing,
        one step later.

        The last case is the other half of the writer's ruling on blanks: a cell is either empty or
        it is not, so a blank arrives as nothing and never as an empty string. Reading it the other
        way would make every ``is null`` check downstream stop matching. Each case is laid out
        beside a second column carrying a key, because a row blank in every column is dropped as
        layout -- the ruling the next test pins -- and a one-column sheet could not tell the two
        rulings apart.
        """
        laid_out = [["value", "key"], *([cell, f"r{index}"] for index, (cell,) in enumerate(cells))]
        target = _workbook(tmp_path / "pack.xlsx", {"Figures": laid_out})

        rows = _read({"path": str(target), "sheet": "Figures"})

        assert rows.schema["value"] == expected_dtype
        assert rows["value"].to_list() == expected_values

    def test_a_row_blank_in_every_column_is_layout_rather_than_a_row(self, tmp_path: Path) -> None:
        """A separator line in a hand-made sheet is not a record of anything.

        The ruling costs something and is still right: a row that is genuinely blank in every column
        carries no value at all, so nothing is lost by dropping it, whereas keeping it puts a
        row of nothing but nulls into every count and every average taken downstream. It is pinned
        because it is a silent decision either way, and somebody comparing the row count in the
        sheet with the row count in the flow needs it written down.
        """
        target = _workbook(
            tmp_path / "pack.xlsx",
            {"Figures": [["key", "amount"], ["r1", 10.25], [None, None], ["r2", -5.5]]},
        )

        rows = _read({"path": str(target), "sheet": "Figures"})

        assert rows["key"].to_list() == ["r1", "r2"]

    @pytest.mark.parametrize(
        ("skip_rows", "outcome"),
        [
            pytest.param(2, "read", id="the_header_is_two_rows_below_the_top"),
            pytest.param(0, "wrong_header", id="and_without_saying_so_the_title_becomes_the_header"),
            pytest.param(-1, "refused", id="a_negative_number_of_rows_is_refused"),
        ],
    )
    def test_a_title_block_above_the_table_is_skipped_to_reach_the_header(
        self,
        tmp_path: Path,
        skip_rows: int,
        outcome: str,
    ) -> None:
        """The one thing about a sheet's layout that cannot be fixed downstream.

        Which row is the header decides every column name in the frame, so it has to be settled by
        the reader; everything else about a cell range can be settled later by tools that already
        exist -- columns to the right of the table are what a column selection is for, and rows
        below it are what a filter is for. That is why this tool takes a count of rows to skip and
        does not take a cell range: a range would be a selection and a filter welded onto a reader,
        expressible one step downstream, where skipping the title block is not.

        The middle case is what happens without it, and it is why the field exists: the report's
        title becomes the first column's name and the real headings become a row of data.
        """
        target = _workbook(
            tmp_path / "pack.xlsx",
            {
                "Figures": [
                    ["Figures for the period", None, None],
                    [None, None, None],
                    ["key", "amount", "code"],
                    ["r1", 10.25, "A"],
                ]
            },
        )
        config = {"path": str(target), "sheet": "Figures", "skip_rows": skip_rows}

        if outcome == "refused":
            with pytest.raises(ConfigError) as caught:
                _read(config)
            assert "skip_rows" in str(caught.value)
            return

        if outcome == "wrong_header":
            with pytest.raises(ConfigError) as caught:
                _read(config)
            assert "header" in str(caught.value).lower(), "a title block leaves two headings blank"
            return

        rows = _read(config)
        assert rows.columns == ["key", "amount", "code"]
        assert rows.row(0) == ("r1", 10.25, "A")

    def test_a_sheet_whose_first_row_is_not_a_header_gets_positional_column_names(
        self,
        tmp_path: Path,
    ) -> None:
        """An extract with no headings at all is real, and the names it gets are pinned by value.

        Positional names are the honest answer to a sheet that names nothing, and they are also a
        warning sign in a configuration: a flow naming ``column_2`` is a flow that breaks when
        somebody inserts a column, so the intended next step is a rename. The names themselves are
        pinned because they end up in the configuration of every step downstream, which makes them
        an interface rather than an implementation detail.
        """
        target = _workbook(tmp_path / "pack.xlsx", {"Figures": [["r1", 10.25], ["r2", -5.5]]})

        rows = _read({"path": str(target), "sheet": "Figures", "has_header": False})

        assert rows.columns == ["column_1", "column_2"]
        assert rows.height == 2


# ---------------------------------------------------------------------------
# Refusals an author can act on
# ---------------------------------------------------------------------------


class TestRefusalsAnAuthorCanActOn:
    """Every way this tool can fail to read, reported in this library's words rather than the reader's."""

    @pytest.mark.parametrize(
        "sheet",
        [
            pytest.param("Summary", id="a_tab_that_is_simply_not_there"),
            pytest.param("figures", id="a_tab_whose_name_differs_only_in_case"),
            pytest.param("Figures ", id="and_one_with_a_trailing_space"),
        ],
    )
    def test_a_tab_that_is_not_in_the_workbook_is_refused_with_the_tabs_that_are(
        self,
        tmp_path: Path,
        sheet: str,
    ) -> None:
        """The refusal is the answer, not merely the complaint.

        A misspelling and a wrong case are the two mistakes anybody makes here, and both are obvious
        the moment the real list of tabs sits beside them -- which is the reasoning the library
        already applies to a column that is not on a port. A near miss is not a match: matching
        loosely would mean a flow configured for ``Figures`` silently reading ``figures ``, and two
        tabs differing only in case are two tabs.
        """
        with pytest.raises(ConfigError) as caught:
            _read({"path": str(_pack(tmp_path / "pack.xlsx")), "sheet": sheet})

        message = str(caught.value)
        assert sheet in message, "the name that was asked for"
        assert "Figures" in message and "Read Me" in message and "Checks" in message, "and the ones that exist"
        assert not isinstance(caught.value, ExecutionError), "a named tab that is absent is not a surprise"

    @pytest.mark.parametrize(
        "situation",
        [
            pytest.param("absent", id="a_file_that_was_never_delivered"),
            pytest.param("not_a_workbook", id="a_file_that_is_not_a_workbook_at_all"),
            pytest.param("empty_tab", id="a_tab_with_nothing_on_it"),
        ],
    )
    def test_a_workbook_that_cannot_be_read_is_refused_naming_the_path(
        self,
        tmp_path: Path,
        situation: str,
    ) -> None:
        """Three failures with one thing in common: the reader underneath explains none of them well.

        Left alone, an absent file arrives as a bare file-not-found with no mention of the flow, a
        file that is not a workbook arrives as a compression error about an end-of-directory record,
        and a tab with nothing on it arrives as an error that helpfully suggests a keyword argument
        of the reader's own -- advice an author of a pipeline definition cannot take, since there is
        nowhere in the definition to put it. So all three are reported here, naming the path.

        The empty tab is refused rather than handed on as a frame with nothing in it because the
        contract refuses an output port with no columns at all: a tab with no header row cannot
        produce a legal port, and refusing here says why, where letting it through would fail one
        step later as a complaint about a column that a downstream step names.
        """
        target = tmp_path / "pack.xlsx"
        sheet = "Figures"
        if situation == "not_a_workbook":
            target.write_text("key,amount\nr1,10.25\n")
        elif situation == "empty_tab":
            _workbook(target, {"Figures": []})

        with pytest.raises(ConfigError) as caught:
            _read({"path": str(target), "sheet": sheet})

        message = str(caught.value)
        assert str(target) in message, "which file, so a delivery can be chased"
        assert not isinstance(caught.value, ExecutionError), "none of these is a surprise inside the tool"
        assert "raise_if_empty" not in message, "the reader's own keyword arguments are not advice an author can take"

    @pytest.mark.parametrize(
        ("header_row", "named"),
        [
            pytest.param(["key", "amount", "key"], "key", id="a_heading_used_twice"),
            pytest.param(["key", None, "code"], "2", id="a_heading_left_blank"),
        ],
    )
    def test_a_header_row_that_repeats_or_omits_a_name_is_refused_rather_than_quietly_repaired(
        self,
        tmp_path: Path,
        header_row: list[Any],
        named: str,
    ) -> None:
        """A sheet's header row is its schema, and a schema that had to be guessed is not one.

        Left to the reader underneath, the second ``key`` becomes ``key_1`` and the blank heading
        becomes a positional placeholder in the reader's own vocabulary. Both are silent, and both
        end the same way: a step downstream names ``key``, gets whichever column happened to come
        first, and the run reports success on the wrong column. Two columns headed the same thing in
        a delivered sheet is a defect in the sheet, and it is cheaper to say so than to pick one.
        """
        target = _workbook(tmp_path / "pack.xlsx", {"Figures": [header_row, ["r1", 10.25, "A"]]})

        with pytest.raises(ConfigError) as caught:
            _read({"path": str(target), "sheet": "Figures"})

        message = str(caught.value)
        assert named in message, "which heading is the problem"
        assert "Figures" in message, "and which tab it is on"

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("extract.csv", id="a_delimited_text_file_is_a_different_tool"),
            pytest.param("extract.xls", id="the_older_spreadsheet_format_is_not_this_one"),
            pytest.param("extract", id="no_extension_at_all"),
        ],
    )
    def test_a_path_that_is_not_an_xlsx_workbook_is_refused_while_there_is_an_author_to_tell(
        self,
        tmp_path: Path,
        name: str,
    ) -> None:
        """One tool, one format, and the refusal says which one this is.

        The two formats differ in their physics rather than their settings -- a text file has no
        tabs, no typed cells and no row limit, and keeps a long integer exactly where a cell cannot
        -- so a single tool covering both would carry a configuration and a set of hazards half of
        which apply to one branch only. Checked against the configuration alone, before anything is
        opened, because that is the moment there is somebody to tell.
        """
        target = tmp_path / name
        target.write_text("key,amount\nr1,10.25\n")

        with pytest.raises(ConfigError) as caught:
            _read({"path": str(target), "sheet": "Figures"})

        assert ".xlsx" in str(caught.value), "the refusal names the format this tool handles"


# ---------------------------------------------------------------------------
# The pair, the template and the catalog
# ---------------------------------------------------------------------------


class TestReadExcelInTheCatalog:
    """The two tools agree with each other, and this one is described and resolvable by name."""

    def test_a_frame_written_by_the_other_tool_reads_back_as_itself(self, tmp_path: Path) -> None:
        """The pair has to agree, and the agreement is a header row and a tab name.

        Everything the writer decides -- that a header is always written, that the tab is the one
        named -- is something the reader has to assume, and neither tool's own tests would notice
        the two drifting apart. This is the test that would: a frame out through one and back through
        the other, values and dtypes compared.
        """
        target = tmp_path / "pack.xlsx"
        frame = pl.DataFrame(
            {
                "key": ["r1", "r2"],
                "amount": [10.25, -5.5],
                "on": [dt.date(2026, 1, 31), dt.date(2026, 2, 28)],
            }
        )

        run_tool(WriteExcel(), Ports({"rows": frame}), {"path": str(target), "sheet": "Figures"})
        back = _read({"path": str(target), "sheet": "Figures"})

        assert_frame_equal(back, frame, check_row_order=True)

    def test_the_template_names_the_tab_and_refuses_to_guess_it(self) -> None:
        """Four fields, one of which is the tab, and deliberately no index and no cell range.

        ``sheet`` is required because the alternative -- defaulting to the first tab -- is a
        positional read that fails silently on the layout a delivered pack actually has, a cover
        sheet first. No index is offered at all, so there is no question of which wins when both are
        given and no way for reordering the tabs to change what a flow reads. And ``path`` is a
        plain string rather than an expression: declared as one, a loader pre-flighting the flow
        would try to compile a filename as a column reference.
        """
        template = ReadExcel().config_template()

        assert set(template) == {"path", "sheet", "sheets_matching", "sheet_column", "has_header", "skip_rows"}
        assert template["path"].required is True
        assert template["path"].non_empty is True
        assert template["sheet"].required is False, "never guessed: naming neither a tab nor a pattern is refused"
        assert template["sheet"].non_empty is True
        assert template["has_header"].type == "boolean"
        assert template["has_header"].default is True, "a spreadsheet's first row is a header until told otherwise"
        assert template["skip_rows"].type == "integer"
        assert template["skip_rows"].default == 0
        assert expression_fields(template) == (), "nothing here is a column reference"

    def test_read_excel_conforms_twice_over_with_the_workbook_it_needs_declared(self) -> None:
        """A source's example needs a world, and it declares one rather than building it in help().

        The workbook an example reads has to exist before the example runs, and the wrong way to
        arrange that is for ``help()`` to create it -- a page generator calls ``help()`` on every
        tool in the catalog, and describing a tool must not write files. So the fixture is declared
        as the example's setup, entered before the example and torn down after it, which is why the
        file it names must be gone by the time this test looks.

        Checked twice on purpose: a setup is a callable returning a context manager exactly so it
        can be entered again, and a version that worked once would pass the first check and fail
        the second on somebody else's run.
        """
        tool = ReadExcel()
        before = sorted(Path.cwd().iterdir())

        check_tool(tool)
        check_tool(tool)

        assert sorted(Path.cwd().iterdir()) == before, "the catalog check must not litter the working directory"
        assert tool.name == "read_excel"
        assert tool.category == "input"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == (), "a source reads no ports; the contract constrains only outputs"
        assert tuple(tool.output_ports) == ("rows",)

        help_ = tool.help()
        for example in help_.examples:
            configured = str(example.configuration["path"])
            assert configured.startswith("{scratch}/"), (
                f"a reader has to be able to copy this straight off the page: {configured}"
            )
            assert example.setup is not None, "the workbook an example reads is a world it declares"

        documented = " ".join((help_.detailed_description, *help_.notes)).lower()
        assert "sheet" in documented or "tab" in documented, "which tab is read has to be documented"
        assert "header" in documented, "and so does what the first row is taken to be"

    def test_read_excel_is_in_the_shipped_catalog_and_the_catalog_still_conforms(self) -> None:
        """Registered under its machine name, and adding it leaves the whole catalog conformant."""
        assert "read_excel" in REGISTRY
        assert isinstance(REGISTRY.get("read_excel"), ReadExcel)

        check_registry(REGISTRY)


# ---------------------------------------------------------------------------
# Every tab whose name matches a pattern: the `sheets_matching` option
#
# A report workbook that gains one tab a month cannot be read by listing its tabs: a list in the
# configuration goes stale the month a tab is added, and a tab nobody listed is a tab nobody reads,
# with no refusal anywhere. `sheets_matching` is the alternative to `sheet`: a pattern, and every
# tab whose name matches it is read and stacked, in workbook order, with `sheet_column` naming the
# tab each row came from.
#
# Rulings pinned here:
#
# - `sheet` and `sheets_matching` are mutually exclusive, and one of them is required. The tab is
#   still never guessed.
# - The pattern is found anywhere in the tab name, like `regex_match()` in an expression, and is
#   case-sensitive. Anchor it with ^ and $ to match whole names.
# - A pattern that matches no tab is refused, with the tabs that are there. It is almost always a
#   mistyped pattern, and an empty result would look like a quiet month.
# - Each matching tab is read exactly as naming it in `sheet` would read it, so every refusal a named
#   tab can draw (a blank or repeated heading, a tab with nothing on it) is drawn by a matched tab
#   too, naming that tab. A tab that does not match is never read.
# - Tabs whose columns differ are stacked by column name. A column one tab lacks is null on that
#   tab's rows; columns come in the order they first appear, and `sheet_column` comes last.
# - Where a column's type differs between tabs: whole and fractional figures stack as fractional
#   figures; a tab where the column holds no values at all takes the other tabs' type; any other
#   difference -- text on one tab, figures on another -- is refused, naming the column and the tabs.
#   Converting figures to text quietly would change what every later step compares them with.
# ---------------------------------------------------------------------------

MONTH_TAB = r"^[A-Z][a-z]{2} \d{4}$"


def _monthly(path: Path) -> Path:
    """A report workbook: a cover tab, three month tabs out of calendar order, and a summary tab.

    The month tabs are deliberately not in alphabetical or calendar order, so a reader that sorted
    the tabs it found would put the rows in a different order from the workbook.
    """
    return _workbook(
        path,
        {
            "Cover": [["Monthly report"]],
            "Mar 2026": [["key", "amount"], ["m1", 1.5], ["m2", 2.5]],
            "Jan 2026": [["key", "amount"], ["j1", 10.25]],
            "Summary": [["total"], [14.25]],
            "Feb 2026": [["key", "amount"], ["f1", -5.5], ["f2", 7.75]],
        },
    )


def _matching(path: Path, pattern: str = MONTH_TAB, **extra: Any) -> pl.DataFrame:
    """Every tab matching ``pattern``, read through the boundary, with the tab named in ``tab``."""
    return _read({"path": str(path), "sheets_matching": pattern, "sheet_column": "tab", **extra})


class TestEveryTabMatchingAPattern:
    """Every matching tab is read, in workbook order, and each row says which tab it came from."""

    def test_every_matching_tab_is_stacked_in_workbook_order_and_no_other_tab_is_read(self, tmp_path: Path) -> None:
        rows = _matching(_monthly(tmp_path / "report.xlsx"))

        expected = pl.DataFrame(
            {
                "key": ["m1", "m2", "j1", "f1", "f2"],
                "amount": [1.5, 2.5, 10.25, -5.5, 7.75],
                "tab": ["Mar 2026", "Mar 2026", "Jan 2026", "Feb 2026", "Feb 2026"],
            }
        )
        assert_frame_equal(rows, expected, check_row_order=True)

    def test_each_tab_stacked_is_exactly_what_naming_that_tab_would_have_read(self, tmp_path: Path) -> None:
        """The option adds tabs and a column. It changes nothing about how one tab is read."""
        path = _monthly(tmp_path / "report.xlsx")

        rows = _matching(path)

        for tab in ("Mar 2026", "Jan 2026", "Feb 2026"):
            alone = _read({"path": str(path), "sheet": tab})
            from_stack = rows.filter(pl.col("tab") == tab).drop("tab")
            assert_frame_equal(from_stack, alone, check_row_order=True)

    def test_the_tab_column_is_text_and_comes_after_every_column_read(self, tmp_path: Path) -> None:
        rows = _matching(_monthly(tmp_path / "report.xlsx"))

        assert rows.columns == ["key", "amount", "tab"]
        assert rows.schema["tab"] == pl.String

    @pytest.mark.parametrize(
        ("pattern", "tabs"),
        [
            pytest.param("2026", ["Jan 2026", "2026 notes"], id="found_anywhere_in_the_name"),
            pytest.param(r"^[A-Z][a-z]{2} \d{4}$", ["Jan 2026"], id="anchors_narrow_it_to_whole_names"),
            pytest.param("(?i)^JAN", ["Jan 2026"], id="case_ignored_only_when_the_pattern_says_so"),
        ],
    )
    def test_the_pattern_is_found_anywhere_in_a_tab_name_unless_anchored(
        self,
        tmp_path: Path,
        pattern: str,
        tabs: list[str],
    ) -> None:
        path = _workbook(
            tmp_path / "report.xlsx",
            {
                "Jan 2026": [["key"], ["j1"]],
                "2026 notes": [["key"], ["n1"]],
                "Summary": [["key"], ["s1"]],
            },
        )

        rows = _matching(path, pattern)

        assert rows["tab"].unique(maintain_order=True).to_list() == tabs

    def test_the_header_and_skip_settings_apply_to_every_matching_tab(self, tmp_path: Path) -> None:
        """A report whose every tab has a title block above the table."""
        titled = [["Report for the month", None], [None, None], ["key", "amount"]]
        path = _workbook(
            tmp_path / "report.xlsx",
            {
                "Jan 2026": [*titled, ["j1", 10.25]],
                "Feb 2026": [*titled, ["f1", -5.5]],
            },
        )

        rows = _matching(path, skip_rows=2)

        assert rows.columns == ["key", "amount", "tab"]
        assert rows["key"].to_list() == ["j1", "f1"]

    def test_tabs_read_without_headers_stack_by_position_and_a_narrower_tab_is_blank_beyond_its_edge(
        self,
        tmp_path: Path,
    ) -> None:
        """The case the option exists for: tabs whose layout moves, read positionally and sorted out later."""
        path = _workbook(
            tmp_path / "report.xlsx",
            {
                "Jan 2026": [["Order", "Cost", "Points"], ["o1", "1.5", "10"]],
                "Feb 2026": [["Order", "Cost"], ["o2", "2.5"]],
            },
        )

        rows = _matching(path, has_header=False)

        assert rows.columns == ["column_1", "column_2", "column_3", "tab"]
        assert rows["column_1"].to_list() == ["Order", "o1", "Order", "o2"]
        assert rows["column_3"].to_list() == ["Points", "10", None, None], "the narrower tab has nothing there"
        assert rows["tab"].to_list() == ["Jan 2026", "Jan 2026", "Feb 2026", "Feb 2026"]

    def test_a_column_one_tab_lacks_is_null_on_that_tabs_rows_and_columns_keep_first_appearance_order(
        self,
        tmp_path: Path,
    ) -> None:
        path = _workbook(
            tmp_path / "report.xlsx",
            {
                "Jan 2026": [["key", "amount", "code"], ["j1", 10.25, "A"]],
                "Feb 2026": [["key", "code", "extra"], ["f1", "B", "x"]],
            },
        )

        rows = _matching(path)

        assert rows.columns == ["key", "amount", "code", "extra", "tab"]
        assert rows["amount"].to_list() == [10.25, None]
        assert rows["extra"].to_list() == [None, "x"]
        assert rows["code"].to_list() == ["A", "B"]

    def test_whole_figures_on_one_tab_and_fractional_on_another_stack_as_figures(self, tmp_path: Path) -> None:
        """A month of round amounts reads back whole; stacked beside a month with cents, both are figures."""
        path = _workbook(
            tmp_path / "report.xlsx",
            {
                "Jan 2026": [["key", "amount"], ["j1", 10.0], ["j2", 20.0]],
                "Feb 2026": [["key", "amount"], ["f1", 2.5]],
            },
        )

        rows = _matching(path)

        assert rows.schema["amount"] == pl.Float64
        assert rows["amount"].to_list() == [10.0, 20.0, 2.5]

    def test_a_column_holding_no_values_on_one_tab_takes_its_type_from_the_others(self, tmp_path: Path) -> None:
        """A month with nothing in a column is not a month of text."""
        path = _workbook(
            tmp_path / "report.xlsx",
            {
                "Jan 2026": [["key", "amount"], ["j1", 10.25]],
                "Feb 2026": [["key", "amount"], ["f1", None]],
            },
        )

        rows = _matching(path)

        assert rows.schema["amount"] == pl.Float64
        assert rows["amount"].to_list() == [10.25, None]

    def test_a_matching_tab_with_a_header_and_no_rows_adds_no_rows_and_refuses_nothing(self, tmp_path: Path) -> None:
        """A month tab set up before the month's figures arrive."""
        path = _workbook(
            tmp_path / "report.xlsx",
            {
                "Jan 2026": [["key", "amount"], ["j1", 10.25]],
                "Feb 2026": [["key", "amount"]],
            },
        )

        rows = _matching(path)

        assert rows["key"].to_list() == ["j1"]
        assert rows.schema["amount"] == pl.Float64


class TestEveryTabMatchingAPatternRefusals:
    """The mistakes the option makes possible, each refused by name."""

    @pytest.mark.parametrize(
        "pattern",
        [
            pytest.param(r"^Dec 2030$", id="a_pattern_no_tab_matches"),
            pytest.param(r"^jan", id="a_pattern_differing_only_in_case"),
        ],
    )
    def test_a_pattern_that_matches_no_tab_is_refused_with_the_tabs_that_are_there(
        self,
        tmp_path: Path,
        pattern: str,
    ) -> None:
        """An empty result would look like a quiet month; it is almost always a mistyped pattern."""
        path = _monthly(tmp_path / "report.xlsx")

        with pytest.raises(ConfigError) as caught:
            _matching(path, pattern)

        message = str(caught.value)
        assert pattern in message, "the pattern that was asked for"
        assert "Jan 2026" in message and "Summary" in message and "Cover" in message, "and the tabs that exist"
        assert str(path) in message, "and which workbook"
        assert not isinstance(caught.value, ExecutionError)

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({"sheet": "Jan 2026", "sheets_matching": MONTH_TAB, "sheet_column": "tab"}, id="both"),
            pytest.param({}, id="neither"),
            pytest.param({"sheet": None, "sheets_matching": None}, id="neither_written_as_null"),
        ],
    )
    def test_naming_a_tab_and_a_pattern_or_neither_is_refused_before_the_workbook_is_opened(
        self,
        tmp_path: Path,
        config: dict[str, Any],
    ) -> None:
        """The workbook does not exist: the refusal must come from the configuration, not the file."""
        absent = tmp_path / "absent.xlsx"

        with pytest.raises(ConfigError) as caught:
            _read({"path": str(absent), **config})

        message = str(caught.value)
        assert "'sheet'" in message and "'sheets_matching'" in message, message
        assert "no file" not in message, "refused on the configuration, before the file is looked for"

    def test_a_pattern_the_engine_cannot_compile_is_refused_before_the_workbook_is_opened(self, tmp_path: Path) -> None:
        absent = tmp_path / "absent.xlsx"

        with pytest.raises(ConfigError) as caught:
            _matching(absent, "Jan (2026")

        message = str(caught.value)
        assert "'sheets_matching'" in message and "Jan (2026" in message
        assert "no file" not in message

    def test_a_pattern_without_a_tab_column_is_refused(self, tmp_path: Path) -> None:
        """Without the tab named on each row, two months' rows cannot be told apart."""
        with pytest.raises(ConfigError) as caught:
            _read({"path": str(tmp_path / "absent.xlsx"), "sheets_matching": MONTH_TAB})

        message = str(caught.value)
        assert "'sheet_column'" in message
        assert "no file" not in message

    def test_a_tab_column_beside_a_single_named_tab_is_refused(self, tmp_path: Path) -> None:
        """It would do nothing, and a field that silently does nothing is a mistake to hear about."""
        with pytest.raises(ConfigError) as caught:
            _read({"path": str(tmp_path / "absent.xlsx"), "sheet": "Jan 2026", "sheet_column": "tab"})

        message = str(caught.value)
        assert "'sheet_column'" in message and "sheets_matching" in message
        assert "unknown" not in message, "the field is known; the refusal is about using it without a pattern"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            pytest.param("sheet_column", "", id="empty_tab_column"),
            pytest.param("sheet_column", "   ", id="whitespace_tab_column"),
            pytest.param("sheets_matching", "", id="empty_pattern"),
        ],
    )
    def test_a_blank_pattern_or_tab_column_is_refused(self, tmp_path: Path, field: str, value: str) -> None:
        path = _monthly(tmp_path / "report.xlsx")
        config = {"path": str(path), "sheets_matching": MONTH_TAB, "sheet_column": "tab", field: value}

        with pytest.raises(ConfigError) as caught:
            _read(config)

        assert f"'{field}'" in str(caught.value)

    def test_a_tab_column_named_like_a_column_on_any_matching_tab_is_refused(self, tmp_path: Path) -> None:
        """Checked on every tab read, not only the first: here only the second tab has the column."""
        path = _workbook(
            tmp_path / "report.xlsx",
            {
                "Jan 2026": [["key"], ["j1"]],
                "Feb 2026": [["key", "tab"], ["f1", "x"]],
            },
        )

        with pytest.raises(ConfigError) as caught:
            _matching(path)

        message = str(caught.value)
        assert "'sheet_column'" in message and "tab" in message

    def test_a_column_holding_text_on_one_tab_and_figures_on_another_is_refused_naming_both_tabs(
        self,
        tmp_path: Path,
    ) -> None:
        """Turning the figures into text quietly would change every later comparison against them."""
        path = _workbook(
            tmp_path / "report.xlsx",
            {
                "Jan 2026": [["key", "amount"], ["j1", 10.25]],
                "Feb 2026": [["key", "amount"], ["f1", "n/a"]],
            },
        )

        with pytest.raises(ConfigError) as caught:
            _matching(path)

        message = str(caught.value)
        assert "amount" in message, "which column"
        assert "Jan 2026" in message and "Feb 2026" in message, "and which tabs disagree"

    def test_a_matching_tab_with_a_bad_header_is_refused_naming_that_tab(self, tmp_path: Path) -> None:
        path = _workbook(
            tmp_path / "report.xlsx",
            {
                "Jan 2026": [["key", "amount"], ["j1", 10.25]],
                "Feb 2026": [["key", "key"], ["f1", "f2"]],
            },
        )

        with pytest.raises(ConfigError) as caught:
            _matching(path)

        message = str(caught.value)
        assert "Feb 2026" in message and "key" in message

    def test_a_tab_that_does_not_match_is_never_read_so_its_bad_header_refuses_nothing(self, tmp_path: Path) -> None:
        """The mirror of the case above."""
        path = _workbook(
            tmp_path / "report.xlsx",
            {
                "Jan 2026": [["key", "amount"], ["j1", 10.25]],
                "Notes": [["key", "key"], [None, "scribble"]],
            },
        )

        rows = _matching(path)

        assert rows["tab"].to_list() == ["Jan 2026"]

    def test_a_matching_tab_with_nothing_on_it_is_refused_naming_the_tab_and_the_workbook(self, tmp_path: Path) -> None:
        path = _workbook(
            tmp_path / "report.xlsx",
            {
                "Jan 2026": [["key", "amount"], ["j1", 10.25]],
                "Feb 2026": [],
            },
        )

        with pytest.raises(ConfigError) as caught:
            _matching(path)

        message = str(caught.value)
        assert "Feb 2026" in message and str(path) in message
        assert not isinstance(caught.value, ExecutionError)


class TestANamedTabIsStillReadAsBefore:
    """The option changes nothing for a flow that names its tab."""

    @pytest.mark.parametrize(
        "extra",
        [
            pytest.param({}, id="new_fields_left_out"),
            pytest.param({"sheets_matching": None, "sheet_column": None}, id="new_fields_written_as_null"),
        ],
    )
    def test_a_named_tab_in_a_workbook_of_many_similar_tabs_reads_that_tab_alone(
        self,
        tmp_path: Path,
        extra: dict[str, Any],
    ) -> None:
        rows = _read({"path": str(_monthly(tmp_path / "report.xlsx")), "sheet": "Feb 2026", **extra})

        expected = pl.DataFrame({"key": ["f1", "f2"], "amount": [-5.5, 7.75]})
        assert_frame_equal(rows, expected, check_row_order=True)


class TestEveryTabMatchingAPatternInTheCatalog:
    """The option is declared in the template and shown in an example that conformance runs."""

    def test_the_template_declares_the_pattern_as_an_alternative_to_the_named_tab(self) -> None:
        template = ReadExcel().config_template()

        assert template["sheet"].required is False, "the pattern is the alternative; one of the two is required"
        assert template["sheet"].default is None
        assert template["sheet"].non_empty is True
        assert template["sheets_matching"].type == "string"
        assert template["sheets_matching"].required is False
        assert template["sheets_matching"].default is None
        assert template["sheets_matching"].non_empty is True
        assert template["sheet_column"].type == "string"
        assert template["sheet_column"].required is False
        assert template["sheet_column"].default is None
        assert expression_fields(template) == (), "a tab pattern is not a column reference"

    def test_an_example_reads_several_matching_tabs_into_one_frame(self) -> None:
        """Conformance executes the example, so the stacking claim is checked in the catalog."""
        tool = ReadExcel()

        check_tool(tool)

        stacked = [example for example in tool.help().examples if example.configuration.get("sheets_matching")]
        assert stacked, "an example must show the option, or the catalog page never mentions it"
        example = stacked[0]
        assert example.configuration.get("sheet") is None
        column = example.configuration["sheet_column"]
        tabs = example.outputs["rows"][column].unique(maintain_order=True).to_list()
        assert len(tabs) >= 2, "the example must stack more than one tab"
        assert example.check_row_order, "workbook order is part of the claim"
        assert example.setup is not None
