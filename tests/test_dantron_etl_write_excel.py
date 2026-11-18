"""Tests for the WriteExcel tool: a frame becomes a sheet, and nothing about it changes silently.

WriteExcel takes one input port, ``rows``, writes it to one sheet of one XLSX workbook, and hands
that same frame back out on ``rows``. It handles XLSX and nothing else: a spreadsheet cell is typed,
a sheet has a row limit, and a workbook has sheet-name rules, none of which are true of a plain text
file -- so a delimited-text sink is a different tool with a different configuration and different
hazards, not a switch on this one.

The rulings pinned here, each argued rather than assumed:

**The sink hands its rows through; it does not emit a receipt.** The contract requires at least one
output port and refuses one with no columns, so a sink has to produce something, and there were two
candidates. A receipt -- one row naming the path, the sheet and the counts -- was the tempting one,
and the argument that decided against it is that every field of it is derivable from the
configuration plus the input frame: the path and the sheet are literally what the author wrote, the
row count is the frame's height, the column count its width. So a receipt adds shape rather than
information, at the price of a second vocabulary of column names on every sink page in the catalog
and of a sink that cannot sit mid-flow without the driver supporting a fan-out. Handing the rows
through costs neither. The one property genuinely lost is that a receipt is one row even when the
input is empty, where a pass-through is zero rows; that is real, and it is outweighed by every sink
in the catalog answering this question the same way.

**An existing file is never overwritten unless the author said so in writing.** A report written to
last month's path is the failure worth preventing, and it is unrecoverable. This is the pattern the
library already uses for a column name that is already in use, so ``replace`` is spelled the same
way and read the same way. The one deliberate difference from ``check_output_column``: ``replace``
on a path that does *not* exist is accepted, because a nightly report's first run writes a new file
and its second overwrites one, and a tool that refused the first run would have to be reconfigured
between two runs of the same flow.

**A promise of exactness is refused, never silently downgraded.** A cell holds a double, so an
integer past 2**53 - 1 cannot come back as itself: 2**53 + 1 is written and read as 2**53, and
123456789012345678 comes back as 123456789012345696. An identifier quietly changing its last digits
is the money-adjacent version of a rounding error, and both alternatives are worse than refusing.
Writing it as text changes the column's type behind the author's back, so a downstream numeric step
breaks somewhere else. Letting it through is the silent corruption itself. So the tool refuses, names
the rows, and the fix is one explicit cast upstream -- the author's decision, in writing, in the
flow. The same argument refuses a ``Decimal`` column, whose entire purpose is exactness a cell cannot
keep. A ``Float64`` is *not* refused, because it is already approximate: it promises nothing the
destination breaks.

**A dtype a cell cannot hold is refused rather than stringified.** A ``Time`` becomes a 1899
datetime, a ``Duration`` becomes a fraction of a day, and a list or a struct becomes a Python repr in
a cell a person will read. Each of those looks like data and is not.

**A frame larger than a sheet is refused, not truncated.** A sheet holds 1,048,576 rows; the header
takes one, so 1,048,575 rows of data fit. Truncation is the worst available outcome for a figure
somebody will add up.

**A frame with no rows writes a header-only sheet, and this is deliberately not what the catalog's
other sink does.** A sink writing to a table refuses an empty frame unless the flow says otherwise,
because replacing a table with nothing empties it. A sink writing to a file has the opposite
obligation: a month with no items still has to be deliverable as a workbook somebody can open, and
a file that was never written is indistinguishable from a run that never happened. The two rulings
differ because the destinations differ, not because the tools disagree.

**Presentation the author did not ask for is not applied.** A date column is given a date format,
because without one a person opening the workbook sees 46053 where the data says 2026-01-31 -- the
format is what makes the value mean what it is, and it is also how a reader identifies the cell as a
date at all. A numeric column is given none: the tool cannot know whether a column is money, a
count or a rate, so any decimal count it chose would be a claim about precision it is not entitled
to make, and a money figure displayed to three decimals is a misstatement in a delivered pack.

Vocabulary is deliberately generic -- key, amount, code -- because the library carries no caller's
nouns.
"""

import datetime as dt
import zipfile
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.tool import (
    CATEGORIES,
    REGISTRY,
    ConfigError,
    DataError,
    Ports,
    check_tool,
    expression_fields,
    run_tool,
)
from dantron_etl.tools.write_excel import (
    MAX_DATA_ROWS,
    MAX_EXACT_INTEGER,
    MAX_SHEET_ROWS,
    WriteExcel,
)

# ---------------------------------------------------------------------------
# Test data and helpers
# ---------------------------------------------------------------------------


def _rows() -> pl.DataFrame:
    """A small frame of the shape a delivered tab actually has."""
    return pl.DataFrame(
        {
            "key": ["r1", "r2", "r3"],
            "amount": [10.25, -5.50, 1234.56],
            "code": ["A", "B", "A"],
        }
    )


def _written(frame: pl.DataFrame, config: Mapping[str, Any]) -> pl.DataFrame:
    """``frame`` put through the boundary, with the single pass-through port unwrapped."""
    result = run_tool(WriteExcel(), Ports({"rows": frame}), dict(config))
    assert result.names == ("rows",), "a sink hands its rows on, so it can sit mid-flow"
    return result["rows"]


def _read_back(path: Path, sheet: str) -> pl.DataFrame:
    """The sheet as the spreadsheet library itself sees it, not as this library's reader sees it.

    Deliberately the raw reader rather than the paired ReadExcel tool: what is under test here is
    the artefact this tool leaves on disk, and reading it back through the other tool would make
    every assertion below depend on that tool's own rulings as well.
    """
    return pl.read_excel(path, sheet_name=sheet, drop_empty_cols=False, raise_if_empty=False)


def _format_codes(path: Path) -> list[str]:
    """Every cell number format the workbook declares -- what a person opening it will see."""
    with zipfile.ZipFile(path) as archive:
        styles = archive.read("xl/styles.xml").decode()
    return [chunk.split('"')[0] for chunk in styles.split('formatCode="')[1:]]


def _parts(path: Path) -> list[str]:
    """The parts inside the workbook, so an invented table object is visible."""
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()


# ---------------------------------------------------------------------------
# What lands in the cells
# ---------------------------------------------------------------------------


class TestWhatReachesTheSheet:
    """The values, the types they come back as, and the places a round trip is lossy."""

    @pytest.mark.parametrize(
        ("column", "expected_dtype", "expected_values"),
        [
            pytest.param(
                pl.Series("amount", [10.25, -5.5, 1234.56]),
                pl.Float64,
                [10.25, -5.5, 1234.56],
                id="a_money_figure_to_two_decimals_is_exact_both_ways",
            ),
            pytest.param(
                pl.Series("count", [1, 0, -7]),
                pl.Int64,
                [1, 0, -7],
                id="a_whole_number_stays_whole",
            ),
            pytest.param(
                pl.Series("code", ["A", "B", "A"]),
                pl.String,
                ["A", "B", "A"],
                id="text_stays_text",
            ),
            pytest.param(
                pl.Series("settled", [True, False, True]),
                pl.Boolean,
                [True, False, True],
                id="a_truth_value_stays_a_truth_value",
            ),
            pytest.param(
                pl.Series("on", [dt.date(2026, 1, 31), dt.date(2026, 2, 28), dt.date(2026, 12, 1)]),
                pl.Date,
                [dt.date(2026, 1, 31), dt.date(2026, 2, 28), dt.date(2026, 12, 1)],
                id="a_date_comes_back_a_date_and_not_a_serial_number",
            ),
            pytest.param(
                pl.Series(
                    "at",
                    [
                        dt.datetime(2026, 1, 31, 13, 45, 30),
                        dt.datetime(2026, 2, 28, 0, 0, 0),
                        dt.datetime(2026, 12, 1, 23, 59, 59),
                    ],
                    dtype=pl.Datetime("us"),
                ),
                pl.Datetime("ms"),
                [
                    dt.datetime(2026, 1, 31, 13, 45, 30),
                    dt.datetime(2026, 2, 28, 0, 0, 0),
                    dt.datetime(2026, 12, 1, 23, 59, 59),
                ],
                id="a_timestamp_keeps_its_instant_and_loses_only_its_resolution",
            ),
            pytest.param(
                pl.Series("amount", [10.25, None, 1234.56]),
                pl.Float64,
                [10.25, None, 1234.56],
                id="a_missing_figure_comes_back_missing",
            ),
            pytest.param(
                pl.Series("note", ["seen", "", None]),
                pl.String,
                ["seen", None, None],
                id="an_empty_string_and_a_missing_value_are_the_same_blank_cell",
            ),
        ],
    )
    def test_a_value_written_to_a_cell_comes_back_as_the_kind_of_thing_it_was(
        self,
        tmp_path: Path,
        column: pl.Series,
        expected_dtype: pl.DataType,
        expected_values: list[Any],
    ) -> None:
        """The whole point of a typed destination: a figure is a figure and a date is a date.

        The date case is the one that catches a real defect. Excel stores a date as a number of
        days, and the cell is only recognisable as a date by the number format written beside it --
        so a writer that skips the format produces a column that reads back as an integer, and every
        date comparison downstream silently compares serial numbers instead.

        The timestamp case pins a change rather than a preservation: a cell has no notion of
        microseconds, so the instant survives and the resolution does not. Written down here because
        a dtype that changes in a sink is otherwise discovered by whoever compares two schemas three
        steps later.

        The last case pins a loss this tool cannot avoid and must not paper over. A cell is either
        empty or it is not; there is no empty-but-present cell, so an empty string comes back as
        nothing. Writing a sentinel instead would put text into a workbook a person reads, and
        reading a blank cell as an empty string would make every ``is null`` check downstream stop
        matching. The loss is one-way, and it is said out loud in the tool's notes.
        """
        target = tmp_path / "pack.xlsx"
        frame = pl.DataFrame({"key": ["r1", "r2", "r3"]}).with_columns(column)

        _written(frame, {"path": str(target), "sheet": "Data"})

        back = _read_back(target, "Data")
        assert back.schema[column.name] == expected_dtype
        assert back[column.name].to_list() == expected_values

    def test_a_frame_with_no_rows_still_writes_its_header_and_hands_the_empty_frame_on(
        self,
        tmp_path: Path,
    ) -> None:
        """A header-only sheet, not an absent file and not an absent sheet.

        Three things could happen to an empty frame and only one of them is defensible here. Writing
        nothing at all leaves whoever opens the pack unable to tell a month with no items from a run
        that never happened. Refusing makes a legitimately empty branch break a flow that is
        otherwise correct -- and AssertRows already exists for a flow that wants emptiness to stop
        the run, said in writing. So the sheet is written with its header, and the empty frame goes
        on through with its schema intact.

        The catalog's table sink rules the other way, and the difference is the destination rather
        than taste: replacing a table with nothing empties the table, where writing a file with
        nothing in it still delivers a workbook somebody can open and see was empty.
        """
        target = tmp_path / "pack.xlsx"
        frame = _rows().head(0)

        passed_on = _written(frame, {"path": str(target), "sheet": "Data"})

        assert passed_on.height == 0
        assert passed_on.schema == frame.schema, "an empty frame still carries its schema onwards"
        back = _read_back(target, "Data")
        assert back.columns == ["key", "amount", "code"], "the header is what says which month was empty"
        assert back.height == 0

    def test_the_workbook_carries_no_presentation_the_author_did_not_ask_for(self, tmp_path: Path) -> None:
        """A date format, because it carries meaning; no number format, because it would invent one.

        Everything a spreadsheet writer can add for free -- banded table styling, an autofilter, a
        thousands separator, red negatives, a fixed decimal count -- is a claim about how the sheet
        will be read, and this tool knows nothing about that. Three decimals on a money column is
        the concrete harm: 1234.56 is delivered reading 1234.560, and a rate column four decimals
        wide is delivered rounded on screen while the cell underneath disagrees with it.
        """
        target = tmp_path / "pack.xlsx"
        frame = pl.DataFrame({"on": [dt.date(2026, 1, 31)], "amount": [1234.56], "count": [7]})

        _written(frame, {"path": str(target), "sheet": "Data"})

        codes = _format_codes(target)
        assert any("yy" in code for code in codes), "without a date format a date cell displays as a number"
        assert not any("0.000" in code for code in codes), "a decimal count is a claim about precision"
        assert not any("Red" in code for code in codes), "how a negative figure is coloured is not this tool's call"
        assert not any("tables/" in part for part in _parts(target)), "a delivered tab is a sheet, not a styled table"


# ---------------------------------------------------------------------------
# What the sink hands on
# ---------------------------------------------------------------------------


class TestWhatComesBackOut:
    """The rows go on unchanged, and the frame that arrived is not touched on the way."""

    def test_the_rows_come_back_out_exactly_as_they_went_in(self, tmp_path: Path) -> None:
        """Values, dtypes and row order, so a sink can sit in the middle of a flow rather than at its end.

        This is what a pass-through has to mean to be worth having. A sink that handed back the
        rows in a different order, or with a column cast to whatever the destination needed, would
        be a transformation wearing a sink's name, and the step after it would be operating on
        something other than what the step before it produced. Row order is checked on purpose: a
        writer that sorted its input to make the sheet tidier would be caught here and nowhere else.
        """
        target = tmp_path / "pack.xlsx"
        frame = _rows()

        passed_on = _written(frame, {"path": str(target), "sheet": "Figures"})

        assert_frame_equal(passed_on, frame, check_row_order=True)

    def test_the_frame_it_was_given_is_left_exactly_as_it_arrived(self, tmp_path: Path) -> None:
        """A sink is the last step, which is the step most tempting to let mutate its input.

        The boundary compares input schemas by itself; the values have to be compared here. A sink
        that cast a column in place to make it writable would corrupt the other branch of any fan-out
        feeding it, and the run would report success.
        """
        target = tmp_path / "pack.xlsx"
        frame = _rows()
        before = frame.clone()

        _written(frame, {"path": str(target), "sheet": "Data"})

        assert frame.equals(before)
        assert frame.schema == before.schema


# ---------------------------------------------------------------------------
# Refusals about the file
# ---------------------------------------------------------------------------


class TestRefusalsAboutTheFile:
    """Where it writes, whether it may, and what it refuses to be asked to write."""

    @pytest.mark.parametrize(
        ("replace", "existing", "outcome"),
        [
            pytest.param(None, True, "refused", id="an_existing_file_is_not_overwritten_by_default"),
            pytest.param(False, True, "refused", id="nor_when_the_author_said_not_to"),
            pytest.param(True, True, "written", id="only_when_the_author_said_so"),
            pytest.param(True, False, "written", id="and_saying_so_still_works_on_the_first_run"),
            pytest.param(None, False, "written", id="a_path_nothing_occupies_needs_no_permission"),
        ],
    )
    def test_an_existing_file_is_overwritten_only_when_the_author_said_so(
        self,
        tmp_path: Path,
        replace: bool | None,
        existing: bool,
        outcome: str,
    ) -> None:
        """The one unrecoverable mistake this tool can make, and the keystroke that authorises it.

        The fourth case is the deliberate difference from the library's rule for a column name
        already in use, which refuses ``replace`` on a name that is not there. A column being
        repaired always exists; a report's path does not exist the first time the flow runs and does
        exist every time after, and the same configuration has to work on both runs. Refusing the
        first run would force an author to edit the flow between two runs of it, which is how a flow
        acquires a second, almost-identical copy of itself.
        """
        target = tmp_path / "pack.xlsx"
        if existing:
            target.write_bytes(b"last month's workbook")
        config: dict[str, Any] = {"path": str(target), "sheet": "Data"}
        if replace is not None:
            config["replace"] = replace

        if outcome == "written":
            _written(_rows(), config)
            assert _read_back(target, "Data").height == 3
            return

        with pytest.raises(ConfigError) as caught:
            _written(_rows(), config)

        message = str(caught.value)
        assert "replace" in message, "the refusal has to name the keystroke that authorises it"
        assert str(target) in message, "and the path it declined to destroy"
        assert target.read_bytes() == b"last month's workbook", "a refused write must not have touched it"

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("report.csv", id="a_delimited_text_file_is_a_different_tool"),
            pytest.param("report.xls", id="the_older_spreadsheet_format_cannot_be_written_at_all"),
            pytest.param("report.xlsm", id="a_macro_workbook_is_not_this"),
            pytest.param("report", id="no_extension_at_all"),
        ],
    )
    def test_a_path_that_is_not_an_xlsx_workbook_is_refused_while_there_is_an_author_to_tell(
        self,
        tmp_path: Path,
        name: str,
    ) -> None:
        """One tool, one format -- and the refusal says which format this one is.

        The formats differ in their physics, not their settings: a text file has no row limit, no
        typed cells and no sheets, and holds a long integer exactly where a cell cannot. A tool
        covering both would carry a configuration and a set of hazards half of which apply to one
        branch. So the extension is checked against the configuration alone, before anything is
        opened, and the refusal names ``.xlsx`` so a reader who arrived at the wrong tool is told
        which one they are holding.
        """
        with pytest.raises(ConfigError) as caught:
            _written(_rows(), {"path": str(tmp_path / name), "sheet": "Data"})

        assert ".xlsx" in str(caught.value), "the refusal names the format this tool handles"
        assert not (tmp_path / name).exists(), "nothing is created by a refusal"

    @pytest.mark.parametrize(
        ("sheet", "named"),
        [
            pytest.param("a" * 32, "31", id="a_name_one_character_too_long"),
            pytest.param("Q1 [draft]", "[", id="a_square_bracket"),
            pytest.param("Assets: current", ":", id="a_colon"),
            pytest.param("rate * 2", "*", id="an_asterisk"),
            pytest.param("balance?", "?", id="a_question_mark"),
            pytest.param("2026/01", "/", id="a_forward_slash"),
            pytest.param("2026\\01", "\\", id="a_back_slash"),
            pytest.param("a" * 31, None, id="but_the_longest_name_allowed_is_accepted"),
        ],
    )
    def test_a_sheet_name_a_workbook_cannot_carry_is_refused_with_the_constraint_named(
        self,
        tmp_path: Path,
        sheet: str,
        named: str | None,
    ) -> None:
        """Refused by this library, in its own words, rather than by the writer underneath.

        A tab name is written by a person for a person, so it attracts exactly the characters a
        workbook forbids -- a date with slashes, a heading with a colon, a draft marker in brackets.
        Left to the writer underneath, the run fails partway through with a message about worksheets;
        left to a writer that silently renames, the pack is delivered with a tab nobody can find by
        the name the flow calls it. The last case is the boundary itself, so the length check is not
        written one character too tight.
        """
        target = tmp_path / "pack.xlsx"

        if named is None:
            _written(_rows(), {"path": str(target), "sheet": sheet})
            assert _read_back(target, sheet).height == 3, "the longest allowed name is a name"
            return

        with pytest.raises(ConfigError) as caught:
            _written(_rows(), {"path": str(target), "sheet": sheet})

        message = str(caught.value)
        assert named in message, "the refusal names the constraint that was broken"
        assert sheet in message or repr(sheet) in message, "and quotes the name that broke it"
        assert not target.exists(), "nothing is created by a refusal"


# ---------------------------------------------------------------------------
# Refusals about the frame
# ---------------------------------------------------------------------------


class TestRefusalsAboutTheFrame:
    """What a cell cannot keep, and the difference between a fault in the rows and one in the shape."""

    @pytest.mark.parametrize(
        ("value", "outcome"),
        [
            pytest.param(MAX_EXACT_INTEGER + 1, "refused", id="one_past_the_largest_exact_whole_number"),
            pytest.param(MAX_EXACT_INTEGER + 2, "refused", id="two_past_it"),
            pytest.param(-(MAX_EXACT_INTEGER + 1), "refused", id="and_the_same_going_the_other_way"),
            pytest.param(123456789012345678, "refused", id="an_identifier_of_the_length_a_real_one_has"),
            pytest.param(MAX_EXACT_INTEGER, "written", id="but_the_largest_exact_whole_number_is_written"),
            pytest.param(-MAX_EXACT_INTEGER, "written", id="and_so_is_its_negative"),
        ],
    )
    def test_a_whole_number_a_cell_cannot_keep_exactly_stops_the_run_as_a_data_fault(
        self,
        tmp_path: Path,
        value: int,
        outcome: str,
    ) -> None:
        """The ruling: refuse it. Neither alternative is honest.

        A cell holds a double, so this value cannot come back as itself -- 123456789012345678 reads
        back 123456789012345696, and an identifier whose last digits changed matches nothing, joins
        to nothing and reconciles to nothing, with no error anywhere. Writing it as text instead
        would change the column's type without being asked, so the break moves to whichever
        downstream step expected a number. Letting it through is the corruption itself. Refusing
        names the rows and leaves the choice with the author: cast it to text one step upstream, on
        purpose, where the next reader can see the decision.

        It is a data fault and not a configuration fault because nothing in the configuration is
        wrong and no edit to it would help; the fix is in the rows, which is where DataError sends
        its reader.

        The two accepted cases are the boundary, so the refusal is not written one off. 2**53 - 1 is
        the largest whole number every double can tell from its neighbour -- 2**53 itself happens to
        be representable while its successor is not, so this is the only boundary that can be
        reasoned about afterwards, and it is the one every other consumer of a spreadsheet already
        uses.
        """
        target = tmp_path / "pack.xlsx"
        frame = pl.DataFrame({"key": ["r1", "r2"], "reference": [1, value]})

        if outcome == "written":
            _written(frame, {"path": str(target), "sheet": "Data"})
            assert _read_back(target, "Data")["reference"].to_list() == [1, value]
            return

        with pytest.raises(DataError) as caught:
            _written(frame, {"path": str(target), "sheet": "Data"})

        message = str(caught.value)
        assert "reference" in message, "which column holds it"
        assert str(value) in message, "and the value itself, so it can be looked up"
        assert not target.exists(), "a refused write creates nothing"

    @pytest.mark.parametrize(
        "column",
        [
            pytest.param(
                pl.Series("amount", [Decimal("1234.56")], dtype=pl.Decimal(12, 2)),
                id="a_figure_whose_whole_purpose_is_exactness",
            ),
            pytest.param(
                pl.Series("at", [dt.time(13, 45)], dtype=pl.Time),
                id="a_time_of_day_becomes_a_date_in_1899",
            ),
            pytest.param(
                pl.Series("took", [dt.timedelta(days=1, hours=2)], dtype=pl.Duration),
                id="a_length_of_time_becomes_a_fraction_of_a_day",
            ),
            pytest.param(
                pl.Series("parts", [[1, 2]], dtype=pl.List(pl.Int64)),
                id="a_list_becomes_a_line_of_punctuation",
            ),
            pytest.param(
                pl.Series("detail", [{"x": 1}], dtype=pl.Struct({"x": pl.Int64})),
                id="and_so_does_a_nested_record",
            ),
            pytest.param(
                pl.Series("at", [dt.datetime(2026, 1, 31, 23, 30)], dtype=pl.Datetime("us", "UTC")),
                id="a_timestamp_with_an_offset_a_cell_cannot_record",
            ),
        ],
    )
    def test_a_column_a_cell_cannot_hold_is_refused_as_a_fault_in_the_shape(
        self,
        tmp_path: Path,
        column: pl.Series,
    ) -> None:
        """Each of these has a silent wrong answer available, which is why none is allowed through.

        A time of day arrives as a datetime in 1899, a duration as 1.083 days, a list or a record as
        a Python repr sitting in a cell somebody will read, a decimal as the double it was chosen to
        avoid, and a timestamp with an offset as a naive local time that no longer says which instant
        it means. Every one of them looks like data.

        Reported as a configuration fault rather than a data fault: no row is more at fault than any
        other, there is nothing to show a reader in the rows, and the fix is authoring -- a step
        upstream that formats the column the way the author wants it to appear. That is the line
        between the two members of the family: DataError when particular rows are at fault and can
        be shown, ConfigError when the shape is.
        """
        target = tmp_path / "pack.xlsx"
        frame = pl.DataFrame({"key": ["r1"]}).with_columns(column)

        with pytest.raises(ConfigError) as caught:
            _written(frame, {"path": str(target), "sheet": "Data"})

        message = str(caught.value)
        assert column.name in message, "which column to go and fix"
        assert str(column.dtype) in message, "and what it currently holds"
        assert not target.exists(), "a refused write creates nothing"

    def test_a_frame_taller_than_a_sheet_is_refused_by_name_rather_than_quietly_truncated(
        self,
        tmp_path: Path,
    ) -> None:
        """The limit is the destination's, and the refusal has to say so.

        Truncation is the worst outcome available here: the pack opens, every figure in it is
        plausible, and the total is wrong by however many rows fell off the end. The check reads the
        frame's height, so it costs nothing and happens before the file is created -- which is also
        how this test pins the rule without a run that writes a million rows.

        A data fault, for the same reason as the oversized integer: no edit to the configuration
        makes it pass, so a reader sent to the file would find nothing to change. What is too big is
        the data.
        """
        target = tmp_path / "pack.xlsx"
        frame = pl.select(pl.int_range(0, MAX_DATA_ROWS + 1, dtype=pl.Int64).alias("key"))

        with pytest.raises(DataError) as caught:
            _written(frame, {"path": str(target), "sheet": "Data"})

        message = str(caught.value)
        assert str(MAX_DATA_ROWS) in message or f"{MAX_DATA_ROWS:,}" in message, "the refusal names the limit"
        assert not target.exists(), "and nothing was written on the way to finding out"

    def test_a_refused_write_leaves_an_existing_workbook_byte_for_byte_as_it_found_it(
        self,
        tmp_path: Path,
    ) -> None:
        """Refusing late is not refusing: a half-written pack is worse than no pack.

        Every refusal about the frame is answerable from the schema or from a column's extremes, so
        every one of them can happen before the workbook is opened. This is the test that keeps it
        that way, and it uses the case with the most to lose -- an authorised overwrite of a file
        that already exists, which a writer that opened the file first would have already truncated
        by the time it noticed the frame was unwritable.
        """
        target = tmp_path / "pack.xlsx"
        target.write_bytes(b"last month's workbook")
        frame = pl.DataFrame({"key": ["r1"], "took": [dt.timedelta(hours=2)]})

        with pytest.raises(ConfigError):
            _written(frame, {"path": str(target), "sheet": "Data", "replace": True})

        assert target.read_bytes() == b"last month's workbook"
        assert sorted(item.name for item in tmp_path.iterdir()) == ["pack.xlsx"], "no scratch file either"


# ---------------------------------------------------------------------------
# The template and the catalog
# ---------------------------------------------------------------------------


class TestWriteExcelInTheCatalog:
    """Described, documented with examples that actually run, and resolvable by name."""

    def test_the_template_says_where_and_which_tab_and_nothing_about_appearance(self) -> None:
        """Three fields, one sheet, and deliberately no presentation and no format switch.

        ``path`` is a plain string and emphatically not an expression: declared as one, a loader
        pre-flighting the flow would try to compile a filename as a column reference. A single
        ``sheet`` field is the whole multi-tab story for now -- a workbook must be written in one
        pass, and a fixed set of declared ports cannot say "N sheets", so a pack of tabs needs
        something the contract does not have yet rather than a fourth field here.

        The limits are published beside the tool so that somebody reading a refusal about one of
        them does not have to open this library's source to check the number.
        """
        template = WriteExcel().config_template()

        assert set(template) == {"path", "sheet", "replace"}
        assert template["path"].type == "string"
        assert template["path"].required is True
        assert template["path"].non_empty is True, "an empty path is a value, and not one to act on"
        assert template["sheet"].required is True, "a tab in a delivered pack is named by a person"
        assert template["sheet"].non_empty is True
        assert template["replace"].type == "boolean"
        assert template["replace"].default is False, "an existing file survives until an author says otherwise"
        assert expression_fields(template) == (), "nothing here is a column reference"

        assert MAX_SHEET_ROWS == 1_048_576
        assert MAX_DATA_ROWS == MAX_SHEET_ROWS - 1, "the header row occupies one of them"
        assert MAX_EXACT_INTEGER == 2**53 - 1

    def test_write_excel_conforms_twice_over_and_leaves_nothing_behind_either_time(self) -> None:
        """Conformance runs every worked example, and a sink's examples really do write.

        Three things are pinned together because they are one property. The check runs **twice**,
        because an example that declares the world it needs declares it as a callable returning a
        context manager precisely so it can be entered again -- a setup good for one run would pass
        the first check and fail the second, which is the shape of defect that only shows up in
        whoever runs the catalog check after you. Nothing may be left in the working directory,
        because an example writing to a relative path drops a workbook wherever the check was run
        from. And the file each example names must be gone afterwards, because a declared world
        that is not torn down is a file on somebody's disk for the rest of the process.
        """
        tool = WriteExcel()
        before = sorted(Path.cwd().iterdir())

        check_tool(tool)
        check_tool(tool)

        assert sorted(Path.cwd().iterdir()) == before, "the catalog check must not litter the working directory"
        assert tool.name == "write_excel"
        assert tool.category == "output"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",), "a pass-through, so a sink can sit mid-flow"

        help_ = tool.help()
        assert len(help_.examples) >= 2, "one plain write, and one that overwrites on purpose"
        for example in help_.examples:
            configured = str(example.configuration["path"])
            assert configured.startswith("{scratch}/"), (
                f"a reader has to be able to copy this straight off the page: {configured}"
            )
            assert example.check_row_order, "a pass-through preserves order, so the example says so"

        documented = " ".join((help_.detailed_description, *help_.notes)).lower()
        assert "sheet" in documented
        assert "overwrit" in documented or "replace" in documented, "the destructive case has to be documented"
        assert "exact" in documented or "precision" in documented, "so does what a cell cannot keep"

    def test_write_excel_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in test_dantron_etl_registry.py."""
        assert "write_excel" in REGISTRY
        assert isinstance(REGISTRY.get("write_excel"), WriteExcel)
