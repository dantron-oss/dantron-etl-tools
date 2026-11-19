"""Tests for the WriteCsv tool: a frame becomes a delimited text file, and the digits survive.

WriteCsv takes one input port, ``rows``, writes it to one delimited text file, and hands that same
frame back out on ``rows``. It handles delimited text and nothing else. The catalog's other file
sink writes XLSX, and the split is not a preference: a sheet has a row limit of 1,048,576, a cell
holds a double and so cannot keep a whole number past 2**53 - 1, a date in a cell is a serial
number that needs a format to mean anything, and a workbook has tab-naming rules. Not one of those
is true here, and the hazards this format has instead -- a value carrying the delimiter, an
encoding nobody declared, a null that cannot be told from an empty string -- are not true there. A
single sink with a format switch would carry a configuration and a page of refusals half of which
apply to one branch only.

The rulings pinned here, each argued rather than assumed:

**The sink hands its rows through; it does not emit a receipt.** The contract requires at least one
output port and refuses one with no columns, so a sink has to produce something. Every field of a
receipt -- the path, the row count, the column count -- is derivable from the configuration plus
the input, so a receipt adds shape rather than information, at the price of a second vocabulary of
column names on every sink page and of a sink that cannot sit mid-flow. The catalog's other sinks
answer this the same way, in the same words, on purpose.

**An existing file is never overwritten unless the author said so in writing**, spelled ``replace``
and defaulting to ``false``. ``overwrite: true`` on a path that holds nothing is accepted, because a
nightly report writes a new file on its first run and overwrites one on every run after, and one
configuration has to serve both runs.

**A whole number is written digit for digit.** This is the sharpest illustration of why this is a
separate tool: 123456789012345678 through a spreadsheet cell comes back 123456789012345696 with
nothing said, and the workbook sink refuses it for that reason. Here it is written as its digits
and is exactly itself, and so is a ``Decimal``, whose whole purpose is an exactness a cell cannot
keep. Nothing about size is refused, and neither is a frame far taller than a sheet.

**A value carrying the delimiter, a quote or a line ending is quoted.** This is where a naive
writer corrupts a file silently: the field becomes two fields, the line becomes one field too
long, and the row still looks like a row. Quoting is the format's own convention -- the field
wrapped in ``"``, an inner quote doubled -- and only the fields that need it are quoted, because a
file a person opens should not be a wall of punctuation.

**The bytes are UTF-8, with no byte order mark, and lines end with a line feed, and none of the
three is configurable.** The paired reader takes an ``encoding`` field because the encoding of a
file somebody else produced is a fact to be declared; a sink is choosing, and there is one choice
worth making. A byte order mark is the close call: a spreadsheet writes one, and without it a
double-clicked file shows accented names wrongly -- but with it, every consumer that does not
special-case three bytes reads the first column's name as something else, silently. The visible
failure is preferable to the silent one, and the note says so.

**A null and an empty string are both written as an empty field**, and the reader hands both back
as nothing. The format cannot tell them apart, and neither convention that could -- a sentinel word
in a file a person reads, or a quoted empty field that almost nothing else honours -- is worth
having. The loss is one-way, it is stated in the tool's own notes, and it is a real reason to
prefer a typed format for anything that matters.

**A frame with no rows writes a header-only file**, because whoever opens it cannot otherwise tell
a month with no items from a run that never happened. The catalog's table sink rules the other way
and the difference is the destination: replacing a table with nothing empties it.

**No refusal leaves a partial or truncated file.** Everything refusable is answerable from the
configuration and the schema, so all of it is refused before the file is opened -- pinned by
re-reading an existing file's bytes after a refused, authorised overwrite.

Vocabulary is deliberately generic -- key, amount, code, note -- because the library carries no
caller's nouns.
"""

import datetime as dt
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.authoring import CATEGORIES, ConfigError, Ports, REGISTRY, Tool, check_tool, run_tool
from dantron_etl.config import expression_fields
from dantron_etl.tools.write_csv import WriteCsv

# A sheet holds 1,048,576 rows and its header takes one of them. Written out rather than imported
# from the workbook sink, because the number is pinned here as a limit this tool does *not* have.
_SHEET_ROWS = 1_048_576


def _rows() -> pl.DataFrame:
    """A small frame of the shape a delivered extract actually has."""
    return pl.DataFrame(
        {
            "key": ["r1", "r2", "r3"],
            "amount": [10.25, -5.50, 1234.56],
            "code": ["A", "B", "A"],
        }
    )


def _written(frame: pl.DataFrame, config: Mapping[str, Any]) -> pl.DataFrame:
    """``frame`` put through the boundary, with the single pass-through port unwrapped."""
    result = run_tool(WriteCsv(), Ports({"rows": frame}), dict(config))
    assert result.names == ("rows",), "a sink hands its rows on, so it can sit mid-flow"
    return result["rows"]


def _one_column(tmp_path: Path, column: pl.Series) -> str:
    """The exact text of a file holding one column, as bytes decoded as UTF-8 and nothing else."""
    target = tmp_path / "extract.csv"
    _written(pl.DataFrame([column]), {"path": str(target)})
    return target.read_bytes().decode()


# ---------------------------------------------------------------------------
# What reaches the file
# ---------------------------------------------------------------------------


class TestWhatReachesTheFile:
    """Every value becomes text, so what that text is, is the whole contract of the sink."""

    @pytest.mark.parametrize(
        ("column", "expected"),
        [
            pytest.param(pl.Series("value", ["A"]), "A", id="text_is_written_as_itself_and_unquoted"),
            pytest.param(pl.Series("value", [10.25]), "10.25", id="a_money_figure_to_two_decimals"),
            pytest.param(pl.Series("value", [-7]), "-7", id="a_whole_number"),
            pytest.param(pl.Series("value", [True]), "true", id="a_truth_value_is_a_word_not_a_number"),
            pytest.param(pl.Series("value", [dt.date(2026, 1, 31)]), "2026-01-31", id="a_date_in_one_order_only"),
            pytest.param(
                pl.Series("value", [dt.datetime(2026, 1, 31, 13, 45, 30)], dtype=pl.Datetime("us")),
                "2026-01-31T13:45:30.000000",
                id="a_timestamp_keeps_every_digit_of_its_resolution",
            ),
            pytest.param(
                pl.Series("value", [dt.datetime(2026, 1, 31, 13, 45, 30)], dtype=pl.Datetime("us", "UTC")),
                "2026-01-31T13:45:30.000000+0000",
                id="and_one_with_an_offset_keeps_the_offset_a_cell_could_not_record",
            ),
            pytest.param(pl.Series("value", [dt.time(13, 45)]), "13:45:00.000000000", id="a_time_of_day"),
            pytest.param(
                pl.Series("value", [Decimal("1234.56")], dtype=pl.Decimal(12, 2)),
                "1234.56",
                id="and_a_figure_whose_whole_purpose_is_exactness_keeps_it",
            ),
            pytest.param(pl.Series("value", [None], dtype=pl.String), "", id="nothing_is_written_as_an_empty_field"),
        ],
    )
    def test_every_kind_of_value_is_written_as_the_text_the_format_says(
        self,
        tmp_path: Path,
        column: pl.Series,
        expected: str,
    ) -> None:
        """A file has no types, so writing is serialisation and every rule of it is visible here.

        Two of these are the reason this tool exists separately from the workbook sink. A
        ``Decimal`` is written as its digits and is exactly itself, where a cell would hold it as a
        double and the workbook sink refuses it outright. A timestamp carrying an offset keeps the
        offset, where a cell has nowhere to record one and the workbook sink refuses that too.

        The others pin the spellings the paired reader has to assume: a truth value is ``true``
        rather than ``TRUE`` or ``1``, a date is written in the one order that cannot be read two
        ways, and a timestamp keeps its microseconds rather than being rounded to the second
        because a sink must not quietly drop resolution.
        """
        assert _one_column(tmp_path, column) == f"value\n{expected}\n"

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(2**53 + 1, id="one_past_the_largest_whole_number_a_cell_could_keep"),
            pytest.param(123456789012345678, id="an_identifier_of_the_length_a_real_one_has"),
            pytest.param(-(2**53 + 1), id="and_the_same_going_the_other_way"),
        ],
    )
    def test_a_long_whole_number_is_written_digit_for_digit_where_a_cell_could_not_keep_it(
        self,
        tmp_path: Path,
        value: int,
    ) -> None:
        """The sharpest illustration of why this is a separate tool rather than a format switch.

        Every one of these values is refused by the workbook sink, and correctly: a cell holds a
        double, so 123456789012345678 would come back 123456789012345696, an identifier whose last
        digits changed matches nothing and reconciles to nothing, and no part of the run would say
        so. A file holds digits, so there is nothing here to refuse and nothing to round. A tool
        with a format switch would have to carry that refusal and then disable it on one branch,
        which is how a refusal ends up applied to the branch it was never about.
        """
        assert _one_column(tmp_path, pl.Series("value", [value])) == f"value\n{value}\n"

    def test_the_bytes_of_the_file_are_exactly_what_the_format_says_and_nothing_more(
        self,
        tmp_path: Path,
    ) -> None:
        """Quoting, the line ending, the mark and the encoding -- all four are properties of the bytes.

        Only the fields that need quoting are quoted, because a file a person opens should not be
        a wall of punctuation, and a writer that quoted nothing would turn the first of these rows
        into two fields, shift every column after it and still produce a file that parses.

        The line ending is a line feed and the file carries no byte order mark, so the same frame
        produces the same bytes wherever the flow runs -- which is what makes two runs of a report
        comparable. The mark is the close call: a spreadsheet writes one, and its absence is why a
        double-clicked file can show an accent wrongly. It is still left out, because its presence
        is the silent failure of the two -- every consumer that does not special-case three bytes
        reads the first column's name as something other than what it is.
        """
        target = tmp_path / "extract.csv"
        frame = pl.DataFrame(
            {
                "key": ["r1", "r2", "r3", "r4"],
                "note": ["holds, a comma", 'holds "a quote"', "holds\na line ending", "café"],
            }
        )

        _written(frame, {"path": str(target)})

        assert (
            target.read_bytes()
            == ('key,note\nr1,"holds, a comma"\nr2,"holds ""a quote"""\nr3,"holds\na line ending"\nr4,café\n').encode()
        )
        assert not target.read_bytes().startswith(b"\xef\xbb\xbf"), "a byte order mark is three bytes of noise"
        assert b"\r\n" not in target.read_bytes(), "one line ending, the same on every machine"

    def test_a_null_and_an_empty_string_are_the_same_empty_field_and_the_loss_is_one_way(
        self,
        tmp_path: Path,
    ) -> None:
        """The convention, both directions, and the fact that it cannot round-trip.

        A file has no way to distinguish an absent value from a present empty one. Both conventions
        that could are worse than the loss: a sentinel word puts text nobody wrote into a file a
        person reads and becomes a real value the moment somebody has a code called ``NULL``, and a
        quoted empty field is honoured by almost nothing that will open this file. So both are
        written as an empty field and the reader hands both back as nothing, which is the same
        ruling the workbook pair makes about a blank cell.

        Pinned here rather than only in the notes because it is the one loss in this pair that a
        reader has to know about before choosing the format: anything where the difference between
        "no note" and "an empty note" matters wants a format with types in it.
        """
        target = tmp_path / "extract.csv"

        _written(pl.DataFrame({"key": ["r1", "r2", "r3"], "note": ["seen", "", None]}), {"path": str(target)})

        assert target.read_text() == "key,note\nr1,seen\nr2,\nr3,\n", "the two are indistinguishable on the page"

    def test_a_frame_with_no_rows_writes_a_header_only_file_and_hands_the_empty_frame_on(
        self,
        tmp_path: Path,
    ) -> None:
        """A header line, not an absent file.

        Three things could happen to an empty frame and only one of them is defensible for a file.
        Writing nothing leaves whoever opens the folder unable to tell a month with no items from a
        run that never happened. Refusing makes a legitimately empty branch break a flow that is
        otherwise correct, and the catalog already has an assertion tool for a flow that wants
        emptiness to stop the run, said in writing.

        The catalog's table sink rules the other way, and the difference is the destination rather
        than taste: replacing a table with nothing empties the table, where writing a file with a
        header still delivers something somebody can open and see was empty.
        """
        target = tmp_path / "extract.csv"
        frame = _rows().head(0)

        passed_on = _written(frame, {"path": str(target)})

        assert target.read_text() == "key,amount,code\n"
        assert passed_on.height == 0
        assert passed_on.schema == frame.schema, "an empty frame still carries its schema onwards"

    def test_a_frame_far_taller_than_a_sheet_is_written_rather_than_refused(self, tmp_path: Path) -> None:
        """The row limit the other sink has is not a limit this one has, and the absence is pinned.

        The workbook sink refuses a frame of more than 1,048,575 rows because a sheet cannot hold
        it and truncation would be the worst available outcome for a figure somebody adds up. A
        text file has no such limit, so a tool with a format switch would either carry the refusal
        onto a branch it was never about or make the limit conditional -- and a conditional limit
        is one somebody eventually applies to the wrong branch. Written out as a real file, because
        a claim that a limit is absent is only worth having if it was actually exceeded.
        """
        target = tmp_path / "extract.csv"
        frame = pl.select(pl.int_range(0, _SHEET_ROWS + 1, dtype=pl.Int64).alias("key"))

        passed_on = _written(frame, {"path": str(target)})

        assert passed_on.height == _SHEET_ROWS + 1
        assert target.read_bytes().count(b"\n") == _SHEET_ROWS + 2, "every row, and the header"


# ---------------------------------------------------------------------------
# What comes back out
# ---------------------------------------------------------------------------


class TestWhatComesBackOut:
    """The rows go on unchanged, and the frame that arrived is not touched on the way."""

    def test_the_rows_come_back_out_exactly_as_they_went_in_and_the_input_is_left_alone(
        self,
        tmp_path: Path,
    ) -> None:
        """What a pass-through has to mean to be worth having, and the mutation it has to exclude.

        A sink that handed back the rows in another order, or with a column cast to whatever the
        destination needed, would be a transformation wearing a sink's name: the step after it
        would be operating on something other than what the step before it produced. Row order is
        checked on purpose, because a writer that sorted its input to make the file tidier would be
        caught here and nowhere else.

        The boundary compares input schemas by itself; the values have to be compared here. A sink
        is the last step in a flow, which is the step most tempting to let mutate its input, and a
        cast in place would corrupt the other branch of any fan-out feeding it while the run
        reported success.
        """
        target = tmp_path / "extract.csv"
        frame = _rows()
        before = frame.clone()

        passed_on = _written(frame, {"path": str(target)})

        assert_frame_equal(passed_on, before, check_row_order=True)
        assert frame.equals(before)
        assert frame.schema == before.schema


# ---------------------------------------------------------------------------
# Refusals, each checked against the file it must not have touched
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
        repaired always exists; a report's path does not exist on the first run of a flow and does
        exist on every run after, and the same configuration has to work on both. Refusing the
        first run would force an author to edit the flow between two runs of it, which is how a
        flow acquires a second, almost-identical copy of itself.

        Answered in the same words as every other sink in the catalog, on purpose: six sink pages
        each explaining overwriting differently is how a reader stops trusting the catalog.
        """
        target = tmp_path / "extract.csv"
        if existing:
            target.write_bytes(b"last month's extract")
        config: dict[str, Any] = {"path": str(target)}
        if replace is not None:
            config["overwrite"] = replace

        if outcome == "written":
            _written(_rows(), config)
            assert target.read_text().startswith("key,amount,code\n")
            return

        with pytest.raises(ConfigError) as caught:
            _written(_rows(), config)

        message = str(caught.value)
        assert "overwrite" in message, "the refusal has to name the keystroke that authorises it"
        assert str(target) in message, "and the path it declined to destroy"
        assert target.read_bytes() == b"last month's extract", "a refused write must not have touched it"

    def test_the_refusal_of_an_existing_file_says_exactly_what_to_add_to_the_step(self, tmp_path: Path) -> None:
        """The second run of a new flow is where this refusal is met, so it says the line to write."""
        target = tmp_path / "extract.csv"
        target.write_bytes(b"last month's extract")

        with pytest.raises(ConfigError) as caught:
            WriteCsv().validate_config({"path": str(target), "delimiter": ",", "overwrite": False})

        assert "add 'overwrite: true'" in str(caught.value)

    @pytest.mark.parametrize("folder", ["missing", "missing/deeper"])
    def test_a_folder_that_is_not_there_is_refused_before_the_flow_runs(self, tmp_path: Path, folder: str) -> None:
        """Found at load, naming the folder, rather than as a bare error after every step has run."""
        target = tmp_path / folder / "extract.csv"

        with pytest.raises(ConfigError) as caught:
            WriteCsv().validate_config({"path": str(target), "delimiter": ",", "overwrite": True})

        message = str(caught.value)
        assert str(target.parent) in message and "'path'" in message, message

    def test_a_path_whose_folder_is_a_file_is_refused_the_same_way(self, tmp_path: Path) -> None:
        (tmp_path / "taken").write_bytes(b"")

        with pytest.raises(ConfigError):
            WriteCsv().validate_config({"path": str(tmp_path / "taken" / "x.csv"), "delimiter": ",", "overwrite": True})

    @pytest.mark.parametrize(
        ("name", "accepted"),
        [
            pytest.param("extract.csv", True, id="the_ordinary_case"),
            pytest.param("extract.tsv", True, id="and_the_name_a_tab_separated_file_is_given"),
            pytest.param("extract.xlsx", False, id="a_workbook_is_the_other_tool"),
            pytest.param("extract.txt", False, id="a_name_that_says_nothing_about_the_format"),
            pytest.param("extract", False, id="and_no_extension_at_all"),
        ],
    )
    def test_a_path_that_is_not_a_delimited_text_file_is_refused_while_there_is_an_author_to_tell(
        self,
        tmp_path: Path,
        name: str,
        accepted: bool,
    ) -> None:
        """One tool, one format, and the refusal says which format this one is.

        Checked against the configuration alone, before anything is opened, because that is the
        moment there is somebody to tell. Two extensions are accepted rather than one: a
        tab-separated file is the same format with a different delimiter, and forcing it to be
        named ``.csv`` would mislead the person who receives it. Everything else is refused, so a
        reader who arrived at the wrong tool of the pair is redirected by name rather than left
        with a workbook full of commas.
        """
        target = tmp_path / name

        if accepted:
            _written(_rows(), {"path": str(target)})
            assert target.exists()
            return

        with pytest.raises(ConfigError) as caught:
            _written(_rows(), {"path": str(target)})

        assert ".csv" in str(caught.value), "the refusal names the format this tool handles"
        assert not target.exists(), "nothing is created by a refusal"

    @pytest.mark.parametrize(
        "column",
        [
            pytest.param(pl.Series("parts", [[1, 2]], dtype=pl.List(pl.Int64)), id="a_list_becomes_punctuation"),
            pytest.param(
                pl.Series("detail", [{"x": 1}], dtype=pl.Struct({"x": pl.Int64})),
                id="and_so_does_a_nested_record",
            ),
            pytest.param(
                pl.Series("took", [dt.timedelta(days=1, hours=2)], dtype=pl.Duration),
                id="and_a_length_of_time_becomes_a_number_of_somethings",
            ),
        ],
    )
    def test_a_column_a_text_file_cannot_carry_is_refused_as_a_fault_in_the_shape(
        self,
        tmp_path: Path,
        column: pl.Series,
    ) -> None:
        """Each of these has a silent wrong answer available, which is why none is allowed through.

        A list or a record written to a field arrives as a line of punctuation that happens to
        contain the delimiter, so the file it produces does not even parse back to the same number
        of columns. A length of time has no agreed spelling at all: written as a count of
        microseconds it reads as a figure somebody will add up.

        Reported as a configuration fault rather than a data fault: no row is more at fault than
        any other, there is nothing to show a reader in the rows, and the fix is authoring -- a
        step upstream that formats the column the way the author wants it to read. That is the line
        between the two members of the family, and it is drawn in the same place here as on every
        other page of the catalog.
        """
        target = tmp_path / "extract.csv"
        frame = pl.DataFrame({"key": ["r1"]}).with_columns(column)

        with pytest.raises(ConfigError) as caught:
            _written(frame, {"path": str(target)})

        message = str(caught.value)
        assert column.name in message, "which column to go and fix"
        assert str(column.dtype) in message, "and what it currently holds"
        assert not target.exists(), "a refused write creates nothing"

    @pytest.mark.parametrize(
        ("delimiter", "accepted"),
        [
            pytest.param(";", True, id="the_delimiter_another_country_expects"),
            pytest.param("\t", True, id="and_the_one_a_tab_separated_file_uses"),
            pytest.param("", False, id="no_delimiter_at_all"),
            pytest.param(";;", False, id="two_characters_are_not_a_delimiter"),
            pytest.param('"', False, id="the_quote_character_cannot_also_separate_fields"),
            pytest.param("\n", False, id="and_neither_can_a_line_ending"),
        ],
    )
    def test_the_delimiter_is_one_character_and_the_author_chooses_which(
        self,
        tmp_path: Path,
        delimiter: str,
        accepted: bool,
    ) -> None:
        """Configurable, because a recipient asks for one; and one character, because more is not one.

        A delimiter of two characters is not a delimiter any consumer of this file agrees with, an
        empty one produces a file with no fields in it, and the quote character or a line ending
        would produce a file that cannot be read back at all -- each of which is a file this tool
        would have written happily and nobody could use. So each is refused, quoting what was
        asked for, before anything is opened.
        """
        target = tmp_path / "extract.csv"
        frame = pl.DataFrame({"key": ["r1"], "note": ["seen"]})

        if accepted:
            _written(frame, {"path": str(target), "delimiter": delimiter})
            assert target.read_text() == f"key{delimiter}note\nr1{delimiter}seen\n"
            return

        with pytest.raises(ConfigError) as caught:
            _written(frame, {"path": str(target), "delimiter": delimiter})

        assert "delimiter" in str(caught.value), "the refusal names the field to fix"
        assert not target.exists(), "nothing is created by a refusal"

    def test_a_refused_write_leaves_an_existing_file_byte_for_byte_as_it_found_it(self, tmp_path: Path) -> None:
        """Refusing late is not refusing: a half-written extract is worse than no extract.

        Every refusal this tool can make is answerable from the configuration and the frame's
        schema, so every one of them can happen before the file is opened. This is the test that
        keeps it that way, and it uses the case with the most to lose -- an authorised overwrite of
        a file that already exists, which a writer that opened its destination first would have
        truncated before it noticed the frame was unwritable.
        """
        target = tmp_path / "extract.csv"
        target.write_bytes(b"last month's extract")
        frame = pl.DataFrame({"key": ["r1"], "took": [dt.timedelta(hours=2)]})

        with pytest.raises(ConfigError):
            _written(frame, {"path": str(target), "overwrite": True})

        assert target.read_bytes() == b"last month's extract"
        assert sorted(item.name for item in tmp_path.iterdir()) == ["extract.csv"], "no scratch file either"


# ---------------------------------------------------------------------------
# The template and the catalog
# ---------------------------------------------------------------------------


class TestWriteCsvInTheCatalog:
    """Described, documented with an example that actually writes, and resolvable by name."""

    def test_the_template_says_where_and_how_and_nothing_a_recipient_cannot_read(self) -> None:
        """Three fields, and five deliberate absences.

        ``path`` is a plain string and emphatically not an expression: declared as one, a loader
        pre-flighting the flow would try to compile a filename as a column reference. There is no
        ``encoding``, because a sink is choosing rather than discovering and there is one choice
        worth making -- the deliberate asymmetry with the paired reader, which must take whatever a
        file somebody else produced was written in. There is no ``quote`` and no ``line_ending``,
        because every non-default value of either produces a file that fewer things can read. There
        is no ``header``, because a file whose column names live only in somebody's memory is not
        one this library will produce. And there is no ``allow_empty``: the table sink needs it
        because a replace of nothing empties a table, where a header-only file is a delivery.
        """
        template = WriteCsv().config_template()

        assert set(template) == {"path", "delimiter", "overwrite"}
        assert template["path"].type == "string"
        assert template["path"].required is True
        assert template["path"].non_empty is True, "an empty path is a value, and not one to act on"
        assert template["delimiter"].default == ","
        assert template["delimiter"].non_empty is True
        assert template["overwrite"].type == "boolean"
        assert template["overwrite"].default is False, "an existing file survives until an author says otherwise"
        assert expression_fields(template) == (), "nothing here is a column reference"

    def test_write_csv_is_a_sink_that_still_hands_its_rows_on(self) -> None:
        """One input, one output of the same name, and a category that says what it is.

        The output port is the ruling every sink in the catalog shares: a pass-through, not a
        receipt. The contract refuses a tool with no output port and refuses an output with no
        columns, so a sink must produce something -- and the only thing it can produce that neither
        loses the data nor invents a second vocabulary is the rows it just wrote.
        """
        tool = WriteCsv()

        assert isinstance(tool, Tool)
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",), "a pass-through, so a sink can sit mid-flow"
        assert tool.category == "output"
        assert tool.category in CATEGORIES

    def test_write_csv_conforms_and_its_example_writes_nothing_into_the_project(self) -> None:
        """Conformance runs every worked example, and a sink's example really does write.

        A sink's example does real I/O, which makes *where* it does it part of the catalog's
        behaviour. An example writing a relative path drops a file wherever the check was run from,
        one writing into the repository shows up as an untracked file after a test run, and one
        that leaves its file behind makes the second run of the same check a different check from
        the first. The example declares its world through ``setup`` rather than having ``help()``
        prepare one, because the catalog page generator calls ``help()`` on every tool and a
        description must not change the process doing the describing.

        Checked twice on purpose: a documented sink that could only be checked once would be a
        documented sink nobody could re-check, which is exactly what a receipt carrying a
        timestamp or a count of what the destination now holds would have made it.
        """
        tool = WriteCsv()
        before = sorted(Path.cwd().iterdir())

        check_tool(tool)
        check_tool(tool)

        assert sorted(Path.cwd().iterdir()) == before, "the catalog check must not litter the working directory"
        assert tool.name == "write_csv"

        help_ = tool.help()
        assert help_.examples, "a sink with no runnable example is a sink nobody can copy"
        for example in help_.examples:
            # No setup assertion: the checker now owns an empty directory per example, so a
            # plain write needs no declared world at all. Only an example that requires a
            # file to *already* be there -- the overwrite case -- still declares one.
            assert example.check_row_order, "a pass-through preserves order, and a reader wants that checked"
            configured = str(example.configuration["path"])
            assert configured.startswith("{scratch}/"), (
                f"a reader has to be able to copy this straight off the page: {configured}"
            )

        documented = " ".join((help_.detailed_description, *help_.notes)).lower()
        assert "overwrit" in documented or "replace" in documented, "the destructive case has to be documented"
        assert "empty" in documented, "and so does the one loss the format cannot avoid"

        assert "write_csv" in REGISTRY
        assert isinstance(REGISTRY.get("write_csv"), WriteCsv)
