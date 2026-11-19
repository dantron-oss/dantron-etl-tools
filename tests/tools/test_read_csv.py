"""Tests for the ReadCsv tool: a delimited text file becomes one frame, inferred and then corrected.

ReadCsv declares **no input ports** -- the contract constrains only ``output_ports``, so a source is
expressible -- and one output port, ``rows``. It handles delimited text and nothing else. The
catalog's other source of a file reads XLSX, and the split is not a preference: a spreadsheet has
tabs, typed cells, a row limit and a 2**53 ceiling on an exact whole number, and a text file has
none of those and has hazards of its own that a spreadsheet does not have. A single tool covering
both would carry a configuration half of whose fields, and half of whose refusals, apply to one
branch only.

The rulings pinned here, each argued rather than assumed:

**A file holds no types, so reading one is inference -- and inference is the default, corrected
per column where it is wrong.** Refusing to infer at all would hand every flow a frame of text and
make the first step after every read a cast; inferring and offering no way to disagree is the
failure this library exists to refuse, because the canonical wrong inference is silent and
irreversible: a column of order numbers reads as a whole number and ``007`` becomes ``7``, which no
step downstream can undo. So ``column_types`` names the columns inference gets wrong, one entry per
column, and everything it does not name is inferred exactly as it would have been.

**Inference reads the whole file, and ``infer_schema_length`` is not a field.** A reader that
decides a column's type from its first hundred lines gives a flow a schema that depends on where in
the file this month's exception happens to sit -- numeric in January, text in February, over rows
nobody changed. A full scan costs one pass and buys a type that is a function of the file. Offering
the limit as a knob would make the failure configurable rather than absent.

**A declared type is named by a word from a published set, spelled the way the query engine spells
it.** ``String``, ``Int64``, ``Float64``, ``Boolean``, ``Date``, ``Datetime``, ``Time`` are the
names a reader already sees in every dtype the catalog prints, so a second vocabulary for the same
seven things would be the two-dialect mistake this library has refused before. A name outside the
set is refused with the set listed beside it.

**A declared date carries the order it is written in.** ``05/01/2026`` is two dates and the file
cannot say which; a reader that guessed would be right for eleven days a month. So an entry may
carry a ``format``, the default is the one spelling that cannot be read two ways, and a format on a
type that is not a date or a time is refused because it would constrain nothing.

**A value that will not convert stops the run**, with the rows named -- and so does a line whose
field count disagrees with the header. Both are the same tension and get the same answer: one bad
line stopping a run of thousands is loud, cheap to fix and recoverable, where the alternatives are
not. A value that would not convert becomes a null, so a column of figures arrives with holes in it
and every total downstream is wrong with no symptom; a short line padded puts nothing where a
figure belongs, and a long one shifts every column after it. There is no threshold and no "ignore
errors" field, because a run that drops some rows silently is the outcome both of these exist to
prevent.

**A blank field is nothing, never an empty string** -- the same ruling the workbook reader makes
about a blank cell, so that an ``is null`` check means one thing across the catalog. A line with
nothing on it is one empty field like any other line, so it is refused in a file of several columns
and is a row of nothing in a file of one: skipping it silently would drop a row the paired writer
can produce.

**A header line that repeats or omits a name is refused, not renamed.** Left to the reader
underneath, a second ``key`` arrives as ``key_duplicated_0`` and a blank heading as a positional
placeholder; a step naming ``key`` then gets whichever came first and nothing says so.

**The bytes are read as the encoding the configuration declares**, and a byte sequence that is not
valid in it is refused rather than read lossily: mojibake is a wrong answer that looks like data. A
byte order mark is accepted and dropped, because that is what a spreadsheet writes and leaving it
makes the first column un-nameable.

**Which of the format's knobs exist.** The delimiter is configurable, because a semicolon file and
a tab file are the same format. The encoding is configurable, because it is a fact about a file
somebody else produced rather than a choice this library gets to make -- the deliberate difference
from the paired writer, which is choosing. The quote character is fixed at ``"`` and both line
endings are accepted, because a knob there would mostly be used to produce files only this library
could read.

Vocabulary is deliberately generic -- key, amount, code, note -- because the library carries no
caller's nouns.
"""

import datetime as dt
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.authoring import (
    CATEGORIES,
    ConfigError,
    DataError,
    ExecutionError,
    Ports,
    REGISTRY,
    Tool,
    ToolError,
    YAML_UNSAFE_VALUES,
    check_tool,
    run_tool,
)
from dantron_etl.config import apply_defaults, expression_fields
from dantron_etl.tools.read_csv import DECLARED_TYPES, ReadCsv
from dantron_etl.tools.write_csv import WriteCsv

# The strings every reader underneath offers as its own way out of the refusals pinned here. None of
# them is advice an author of a pipeline definition can take: there is nowhere in a definition to
# put a keyword argument of a library it has never heard of.
_READER_KEYWORDS = ("truncate_ragged_lines", "raise_if_empty", "ignore_errors", "infer_schema_length")


def _file(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write ``text`` to ``path`` byte for byte -- no line-ending translation, no mark, nothing added."""
    path.write_bytes(text.encode(encoding))
    return path


def _read(config: Mapping[str, Any]) -> pl.DataFrame:
    """``config`` put through the boundary with no input ports at all, and ``rows`` unwrapped."""
    result = run_tool(ReadCsv(), Ports(), dict(config))
    assert result.names == ("rows",), "a source has one output port and no inputs"
    return result["rows"]


def _laid_out(tmp_path: Path, text: str, **config: Any) -> pl.DataFrame:
    """The frame ``text`` reads back as, written to a file first because that is what a source reads."""
    return _read({"path": str(_file(tmp_path / "extract.csv", text)), **config})


# ---------------------------------------------------------------------------
# What a column comes back as: inference, and the entry that corrects it
# ---------------------------------------------------------------------------


class TestWhatAColumnComesBackAs:
    """Inferred from the whole file, and exactly what was declared for the columns that say so."""

    @pytest.mark.parametrize(
        ("written", "expected_dtype", "expected"),
        [
            pytest.param(["10.25", "-5.5"], pl.Float64, [10.25, -5.5], id="a_column_of_figures_is_figures"),
            pytest.param(["1", "-7"], pl.Int64, [1, -7], id="a_column_of_whole_numbers_is_whole_numbers"),
            pytest.param(["true", "false"], pl.Boolean, [True, False], id="and_two_words_are_a_truth_value"),
            pytest.param(
                ["123456789012345678", "9007199254740993"],
                pl.Int64,
                [123456789012345678, 9007199254740993],
                id="a_long_identifier_is_exact_where_a_spreadsheet_cell_could_not_be",
            ),
            pytest.param(
                ["2026-01-31", "2026-02-28"],
                pl.String,
                ["2026-01-31", "2026-02-28"],
                id="but_a_date_is_text_until_somebody_declares_it",
            ),
            pytest.param(
                ["0", "0.5", "-0.25", "10"],
                pl.Float64,
                [0.0, 0.5, -0.25, 10.0],
                id="a_zero_before_the_point_is_not_a_leading_zero",
            ),
            pytest.param(
                ["9223372036854775807", "-9223372036854775808"],
                pl.Int64,
                [9223372036854775807, -9223372036854775808],
                id="the_largest_whole_numbers_64_bits_hold_are_still_whole_numbers",
            ),
            pytest.param(
                ["007", "A12"],
                pl.String,
                ["007", "A12"],
                id="a_code_column_that_is_not_all_numbers_is_text_zeros_and_all",
            ),
            pytest.param(["", ""], pl.String, [None, None], id="a_column_of_blanks_is_text_holding_nothing"),
            pytest.param(
                ["1"] * 200 + ["not a number"],
                pl.String,
                ["1"] * 200 + ["not a number"],
                id="and_one_line_in_two_hundred_decides_the_column_because_the_whole_file_is_read",
            ),
        ],
    )
    def test_a_column_nobody_declared_is_inferred_and_two_of_those_inferences_are_traps(
        self,
        tmp_path: Path,
        written: Sequence[str],
        expected_dtype: pl.DataType,
        expected: Sequence[Any],
    ) -> None:
        """Inference is the default because it is right most of the time; these are the edges of it.

        A column of codes that is not all numbers stays text, zeros and all. A column that is all
        numbers but carries a leading zero is refused rather than read; that is the next test.

        The date case is the other: a date in a file is text, because deciding that a column of
        eight digits and two separators is a date -- and in which order -- is a guess the file
        cannot settle. So it stays text, and an author who wants dates says so and says in which
        order they are written.

        The long identifier is the contrast with the catalog's workbook reader that justifies two
        tools: a cell holds a double, so it cannot tell 9007199254740993 from its neighbour, and a
        file holds digits, so this one comes back exactly.

        The last case is the ruling about how much of the file inference reads: all of it. Every
        reader underneath samples the first hundred lines by default, and a column that is numeric
        for two hundred lines and carries one reference at the end is exactly the shape that
        defeats a sample -- the read either fails on the last line or hands that value back as a
        null in a numeric column. The deeper problem is that the column's type would then depend
        on where in the file this month's exception happens to sit, so a definition that worked in
        January fails in February over rows nobody changed. A full scan costs one pass, and
        offering the sample size as a field would make that failure configurable rather than
        absent.
        """
        rows = _laid_out(tmp_path, "value\n" + "".join(f"{value}\n" for value in written))

        assert rows.schema["value"] == expected_dtype
        assert rows["value"].to_list() == list(expected)
        assert rows.height == len(written), "and nothing is dropped on the way to deciding"

    @pytest.mark.parametrize(
        ("written", "line", "offender"),
        [
            pytest.param(["007", "010"], 2, "007", id="an_order_number_with_its_zeros"),
            pytest.param(["12", "-007"], 3, "-007", id="a_signed_one"),
            pytest.param(["1.5", "00.5"], 3, "00.5", id="a_figure_written_with_a_zero_in_front"),
            pytest.param([str(n) for n in range(1, 201)] + ["02134"], 202, "02134", id="one_line_in_two_hundred"),
            pytest.param(["12345678901234567890", "7"], 2, "12345678901234567890", id="a_whole_number_past_64_bits"),
            pytest.param(["1.5", "-99999999999999999999"], 3, "-99999999999999999999", id="beside_figures_too"),
        ],
    )
    def test_a_column_a_number_would_change_is_refused_rather_than_guessed(
        self,
        tmp_path: Path,
        written: Sequence[str],
        line: int,
        offender: str,
    ) -> None:
        """Inference stops where reading a column as numbers would change what the file says.

        A leading zero is dropped by a number, and a whole number past 64 bits can only be read as a
        rounded figure. Either way the value no longer matches the file, and nothing downstream can
        get it back. So the read stops and names the column, the line and the value, and says what
        to write in column_types.
        """
        with pytest.raises(DataError) as caught:
            _laid_out(tmp_path, "value\n" + "".join(f"{value}\n" for value in written))

        message = str(caught.value)
        assert "'value'" in message, f"names the column: {message}"
        assert offender in message, f"names the value: {message}"
        assert f"line {line}" in message, f"names the line: {message}"
        assert "column_types" in message and "String" in message, f"says what to write: {message}"

    @pytest.mark.parametrize(
        ("declared", "written", "expected_dtype", "expected"),
        [
            pytest.param("String", "007", pl.String, "007", id="text_keeps_the_zeros"),
            pytest.param("Int64", "007", pl.Int64, 7, id="a_number_drops_them_because_the_author_said_so"),
            pytest.param("String", "12345678901234567890", pl.String, "12345678901234567890", id="text_keeps_digits"),
            pytest.param("Float64", "12345678901234567890", pl.Float64, 1.2345678901234567e19, id="a_figure_rounds"),
        ],
    )
    def test_a_declared_column_is_never_refused_for_what_a_number_would_change(
        self,
        tmp_path: Path,
        declared: str,
        written: str,
        expected_dtype: pl.DataType,
        expected: Any,
    ) -> None:
        """The refusal is about guessing. Once the author states the type, the read does as told."""
        rows = _laid_out(
            tmp_path, f"value\n{written}\n", column_types=[{"column": "value", "type": declared}]
        )

        assert rows.schema["value"] == expected_dtype
        assert rows["value"].to_list() == [expected]

    @pytest.mark.parametrize(
        ("entry", "written", "expected_dtype", "expected"),
        [
            pytest.param({"type": "String"}, "007", pl.String, "007", id="the_zeros_an_inference_would_have_eaten"),
            pytest.param({"type": "Int64"}, "42", pl.Int64, 42, id="a_whole_number_declared_rather_than_guessed"),
            pytest.param({"type": "Float64"}, "10", pl.Float64, 10.0, id="a_rate_that_happens_to_be_round_this_month"),
            pytest.param({"type": "Boolean"}, "true", pl.Boolean, True, id="a_truth_value_as_the_writer_spells_it"),
            pytest.param(
                {"type": "Date"},
                "2026-01-31",
                pl.Date,
                dt.date(2026, 1, 31),
                id="a_date_in_the_one_order_that_cannot_be_read_two_ways",
            ),
            pytest.param(
                {"type": "Date", "format": "%d/%m/%Y"},
                "05/01/2026",
                pl.Date,
                dt.date(2026, 1, 5),
                id="and_one_whose_order_the_author_had_to_say_out_loud",
            ),
            pytest.param(
                {"type": "Datetime"},
                "2026-01-31T13:45:30.000000",
                pl.Datetime("us"),
                dt.datetime(2026, 1, 31, 13, 45, 30),
                id="a_timestamp",
            ),
            pytest.param({"type": "Time"}, "13:45:00.000000000", pl.Time, dt.time(13, 45), id="a_time_of_day"),
            pytest.param({"type": "Int64"}, "", pl.Int64, None, id="and_a_blank_is_nothing_whatever_was_declared"),
        ],
    )
    def test_a_declared_column_is_read_as_declared_and_its_neighbour_is_still_inferred(
        self,
        tmp_path: Path,
        entry: Mapping[str, str],
        written: str,
        expected_dtype: pl.DataType,
        expected: Any,
    ) -> None:
        """What an entry buys, and the property that makes it an override rather than a schema.

        ``key`` is never declared in any of these cases and holds whole numbers, so it must come
        back a whole number every time. An implementation that applied the declared type to the
        frame rather than to the column named is the obvious wrong one, and this is where it is
        caught -- from the outside there is no other difference between the two.

        The two date cases are the ruling about format. Given none, one spelling is accepted, and
        it is the one that cannot be read two ways. Given a format, the author has said in writing
        which of the day and the month comes first, which is the only way that question can be
        answered about a file: ``05/01/2026`` is two dates, and a reader that guessed would be
        right for eleven days of every month and wrong for the rest with nothing said.

        The last case pins that a blank field is nothing in a declared column too -- not a zero,
        and not a conversion failure.
        """
        rows = _laid_out(
            tmp_path,
            f"key,value\n42,{written}\n",
            column_types=[{"column": "value", **entry}],
        )

        assert rows.schema["value"] == expected_dtype
        assert rows["value"].to_list() == [expected]
        assert rows.schema["key"] == pl.Int64, "a column nobody named is inferred exactly as it would have been"

    @pytest.mark.parametrize(
        ("entry", "written"),
        [
            pytest.param({"type": "Int64"}, "abc", id="a_word_where_a_whole_number_was_declared"),
            pytest.param({"type": "Int64"}, "10.5", id="a_figure_with_a_fraction_is_not_a_whole_number"),
            pytest.param({"type": "Float64"}, "1 234.56", id="a_figure_carrying_a_separator"),
            pytest.param({"type": "Boolean"}, "yes", id="a_truth_value_spelled_another_way"),
            pytest.param({"type": "Date"}, "31/01/2026", id="a_date_in_an_order_nobody_declared"),
            pytest.param(
                {"type": "Date", "format": "%d/%m/%Y"},
                "2026-01-31",
                id="and_one_in_the_order_the_declaration_says_it_is_not",
            ),
        ],
    )
    def test_a_value_that_will_not_convert_to_its_declared_type_stops_the_run_rather_than_reading_as_nothing(
        self,
        tmp_path: Path,
        entry: Mapping[str, str],
        written: str,
    ) -> None:
        """The behaviour of every reader underneath is the one outcome that must not happen.

        Left alone, each of these arrives as a null: the column keeps its declared dtype, the frame
        looks right, and the values that could not be converted are simply gone. A total taken
        downstream is then wrong by however many rows failed, and no part of the run mentions it.
        Refusing makes the same fault loud and names the values, which is the only version of this
        an author can act on -- and the fix is theirs to choose, because the declaration may be
        wrong or the file may be.

        The same answer as a line whose field count disagrees with the header, and deliberately so:
        both stop a run of thousands over a handful of values, and in both the alternative is a
        frame that looks right and is not. There is no threshold field and no way to ask for the
        nulls, because a run that drops rows quietly is what this refusal exists to prevent.

        A fault in the rows rather than in the configuration: nothing about the file's declaration
        is malformed, and the reader's next move is to go and look at the values, which is where
        ``DataError`` sends them.
        """
        with pytest.raises(DataError) as caught:
            _laid_out(tmp_path, f"key,value\n42,{written}\n", column_types=[{"column": "value", **entry}])

        message = str(caught.value)
        assert "value" in message, "which column to look at"
        assert written in message, "and the value that would otherwise have become a null"
        assert entry["type"] in message, "and what it was declared to be"

    @pytest.mark.parametrize(
        ("column_types", "named"),
        [
            pytest.param(
                [{"column": "total", "type": "Float64"}],
                ("total", "key", "amount"),
                id="a_column_the_header_has_not_got_is_shown_the_header",
            ),
            pytest.param(
                [{"column": "amount", "type": "Float"}],
                ("Float", "Float64"),
                id="a_type_name_outside_the_published_set_is_shown_the_set",
            ),
            pytest.param(
                [{"column": "amount", "type": "Boolean", "format": "%d/%m/%Y"}],
                ("format", "Date"),
                id="a_format_on_a_type_no_format_says_anything_about",
            ),
            pytest.param(
                [{"column": "amount", "type": "Float64"}, {"column": "amount", "type": "String"}],
                ("amount",),
                id="and_two_rulings_for_one_column",
            ),
        ],
    )
    def test_a_declaration_the_tool_cannot_honour_is_refused_with_the_alternatives_beside_it(
        self,
        tmp_path: Path,
        column_types: Sequence[Mapping[str, str]],
        named: tuple[str, ...],
    ) -> None:
        """Four authoring mistakes, each refused with the thing that was actually available.

        A misspelled column is the one that most needs saying: unchecked, the column the author
        meant keeps being inferred, every step still runs, and the declaration they wrote does
        nothing at all. Listing the header beside it is what turns the refusal into the answer, in
        the same shape the library already uses for a column that is not on a port.

        A type name outside the published set fails closed rather than being ignored, and the set
        is listed, because the mistake is a spelling and the list is the fix. The published
        spellings are the query engine's own -- what a reader already sees printed in every dtype
        in this catalog -- rather than a second vocabulary invented for the same seven things.
        """
        with pytest.raises(ConfigError) as caught:
            _laid_out(tmp_path, "key,amount\nr1,10.25\n", column_types=list(column_types))

        message = str(caught.value)
        for word in named:
            assert word in message, f"the refusal should mention {word!r}: {message}"


# ---------------------------------------------------------------------------
# What the text itself can carry
# ---------------------------------------------------------------------------


class TestWhatTheTextItselfCanCarry:
    """The characters that make a delimited file hard, and the blank that makes it lossy."""

    @pytest.mark.parametrize(
        ("delimiter", "written", "expected"),
        [
            pytest.param(",", '"holds, a comma"', "holds, a comma", id="a_field_carrying_the_delimiter_is_one_field"),
            pytest.param(",", '"holds ""a quote"""', 'holds "a quote"', id="a_doubled_quote_is_one_quote"),
            pytest.param(",", '"holds\na line ending"', "holds\na line ending", id="a_field_may_span_two_lines"),
            pytest.param(";", "holds, a comma", "holds, a comma", id="and_under_another_delimiter_a_comma_is_ordinary"),
            pytest.param("\t", '"holds\ta tab"', "holds\ta tab", id="a_tab_file_quotes_its_own_delimiter"),
        ],
    )
    def test_a_field_carrying_the_delimiter_a_quote_or_a_line_ending_arrives_whole(
        self,
        tmp_path: Path,
        delimiter: str,
        written: str,
        expected: str,
    ) -> None:
        """This is where a naive reader silently corrupts a file, so it is pinned by value.

        A field holding the delimiter read naively becomes two fields, which makes the line one
        field too long and shifts every column after it -- and the row still looks like a row. The
        quoting convention is the format's own (a field is wrapped in ``"``, and a quote inside it
        is doubled), it is not configurable, and it is what the paired writer produces.

        The fourth case is the reason the delimiter is a field at all: under a semicolon, a comma
        is an ordinary character, which is exactly the file a recipient in another country sends.
        """
        rows = _laid_out(tmp_path, f"key{delimiter}note\nr1{delimiter}{written}\n", delimiter=delimiter)

        assert rows["note"].to_list() == [expected]
        assert rows.height == 1, "a quoted line ending is part of the field, not the end of the row"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param("key,note\nr1,\n", [("r1", None)], id="a_blank_field_is_nothing_rather_than_an_empty_string"),
            pytest.param('key,note\nr1,""\n', [("r1", None)], id="and_so_is_a_quoted_blank_field"),
            pytest.param("key,note\n,\n", [(None, None)], id="a_line_of_blank_fields_is_still_a_row"),
            pytest.param("key,note\nr1,seen", [("r1", "seen")], id="a_last_line_with_no_line_ending_is_a_row"),
            pytest.param("key,note\nr1,seen\n", [("r1", "seen")], id="and_the_trailing_line_ending_adds_none"),
        ],
    )
    def test_a_blank_field_is_nothing_and_a_line_of_blank_fields_is_still_a_row(
        self,
        tmp_path: Path,
        text: str,
        expected: Sequence[tuple[Any, ...]],
    ) -> None:
        """Two rulings that have to be settled together, because the format cannot tell them apart.

        A blank field reads as nothing, which matches the workbook reader's blank cell, so an ``is
        null`` check means the same thing on both pages of the catalog. The consequence is the loss
        the paired writer's notes state out loud: an empty string written out comes back as
        nothing, and no convention available here avoids that without putting a sentinel word into
        a file a person reads.

        A line whose fields are all blank is still a row, because the paired writer produces
        exactly that line for a row of nulls. Dropping it as layout -- which is what the workbook
        reader does with a row of empty cells, and which reads as reasonable until the pair is put
        together -- would make a row disappear on the way through a round trip.
        """
        rows = _laid_out(tmp_path, text)

        assert rows.rows() == list(expected)

    @pytest.mark.parametrize(
        ("text", "line", "fields"),
        [
            pytest.param("key,note,code\nr1,seen,A\nr2,seen\n", 3, 2, id="a_short_line_would_be_padded_with_nothing"),
            pytest.param("key,note,code\nr1,seen,A\nr2,seen,B,C\n", 3, 4, id="a_long_line_would_shift_every_column"),
            pytest.param("key,note,code\nr1,seen,A\n\nr2,seen,B\n", 3, 1, id="and_a_line_with_nothing_on_it_is_short"),
        ],
    )
    def test_a_line_whose_field_count_the_header_does_not_agree_with_is_refused_naming_the_line(
        self,
        tmp_path: Path,
        text: str,
        line: int,
        fields: int,
    ) -> None:
        """Refuse, rather than pad or truncate, and say which line to open.

        Padding is the silent one: the short line arrives with nulls in its last columns, so a
        figure is missing rather than wrong, and the run reports success. Truncating a long line
        loses a field outright. Both produce a frame nobody questions -- so one bad line stops the
        run, and the refusal carries the line number, the count it found and the count it expected,
        which is everything needed to open the file and look.

        The same answer as a value that will not convert to its declared type, for the same reason
        and with the same cost: a run of thousands stops over one line. A fault in the rows and not
        in the configuration, because no edit to the definition makes a malformed line well formed.

        The refusal is this library's own. Left to the reader underneath, a long line arrives as a
        complaint that recommends one of its own keyword arguments, which is advice an author of a
        pipeline definition has nowhere to put.
        """
        with pytest.raises(DataError) as caught:
            _laid_out(tmp_path, text)

        message = str(caught.value)
        assert "line" in message.lower(), "which line to open"
        assert str(line) in message, "and its number"
        assert str(fields) in message, "and how many fields it actually had"
        assert not any(keyword in message for keyword in _READER_KEYWORDS), (
            f"a reader's own keyword arguments are not advice an author can take: {message}"
        )


# ---------------------------------------------------------------------------
# The bytes, and the shape of the file around the header
# ---------------------------------------------------------------------------


class TestTheBytesAndTheShapeOfTheFile:
    """What the file is decoded as, and where the header is found in it."""

    @pytest.mark.parametrize(
        ("raw", "encoding", "expected"),
        [
            pytest.param(
                "key,note\nr1,café\n".encode("latin-1"),
                "latin1",
                "café",
                id="the_encoding_it_was_written_in",
            ),
            pytest.param("key,note\nr1,don’t\n".encode("cp1252"), "cp1252", "don’t", id="a_word_processor_s_own_page"),
            pytest.param("key,note\nr1,café\n".encode(), "utf8", "café", id="and_the_one_almost_everything_uses"),
            pytest.param("﻿key,note\nr1,café\n".encode(), "utf8", "café", id="a_mark_a_spreadsheet_left_behind"),
            pytest.param("key,note\r\nr1,café\r\n".encode(), "utf8", "café", id="and_the_other_half_s_line_ending"),
            pytest.param(
                "key,note\nr1,café\n".encode("latin-1"),
                "utf8",
                None,
                id="and_bytes_that_are_not_valid_in_it_are_refused_rather_than_mangled",
            ),
        ],
    )
    def test_the_bytes_are_read_as_the_declared_encoding_and_the_file_s_punctuation_never_reaches_a_value(
        self,
        tmp_path: Path,
        raw: bytes,
        encoding: str,
        expected: str | None,
    ) -> None:
        """Four separate ways a value can arrive carrying something that is not part of it.

        The encoding is a fact about a file somebody else produced, so it is configurable here and
        fixed in the paired writer, which is choosing rather than discovering. A byte order mark is
        three bytes a spreadsheet writes and nothing else expects; left in place it becomes part of
        the first column's name, so a step naming ``key`` finds no such column and the refusal it
        gets names a column that looks identical to the one it asked for. A carriage return left on
        the end of the last field is the same defect one column further right: every comparison
        against it fails, and the value prints as though it matched.

        The last case is the one every reader underneath offers a way around, and this tool does
        not pass it on. Read lossily, a byte that is not valid in the declared encoding becomes a
        replacement character, and the column arrives full of plausible-looking text that is not
        what anybody wrote -- a name that no longer matches, a code that no longer joins. So it is
        refused, naming the file, the encoding it was read as, and where the reading stopped.
        """
        target = tmp_path / "extract.csv"
        target.write_bytes(raw)
        # ``key`` is declared on purpose: a mark or a carriage return left anywhere near the header
        # makes the column this names unfindable, which is the same defect one step downstream.
        config = {"path": str(target), "encoding": encoding, "column_types": [{"column": "key", "type": "String"}]}

        if expected is None:
            with pytest.raises(ConfigError) as caught:
                _read(config)
            message = str(caught.value)
            assert str(target) in message, "which file"
            assert encoding in message, "and what it was read as"
            assert "�" not in message, "a refusal about mojibake must not itself contain mojibake"
            return

        rows = _read(config)

        assert rows.columns == ["key", "note"], "no mark on the first column's name"
        assert rows["note"].to_list() == [expected]
        assert rows["key"].to_list() == ["r1"], "and no carriage return on the end of a value"

    @pytest.mark.parametrize(
        ("config", "outcome"),
        [
            pytest.param({"skip_rows": 2}, "read", id="the_header_is_two_lines_below_the_top"),
            pytest.param({}, "refused", id="and_without_saying_so_the_title_line_becomes_the_header"),
            pytest.param({"skip_rows": -1}, "refused_config", id="a_negative_number_of_lines_is_refused"),
        ],
    )
    def test_a_title_block_above_the_header_is_skipped_to_reach_it(
        self,
        tmp_path: Path,
        config: Mapping[str, Any],
        outcome: str,
    ) -> None:
        """Which line is the header decides every column name, so it is settled by the reader.

        Everything else about the shape of an extract can be settled one step downstream by tools
        that already exist -- columns nobody wants are a column selection, lines below the table
        are a filter -- so this is the only such field the tool needs, and the workbook reader
        carries the same one for the same reason.

        The middle case is what the field prevents, and the way it fails is particular to this
        format: the title line has one field and the real header has three, so every line below it
        disagrees with the header the tool was left to infer. Refused, rather than read.
        """
        text = "Figures for the period\n\nkey,amount,code\nr1,10.25,A\n"

        if outcome == "read":
            rows = _laid_out(tmp_path, text, **config)
            assert rows.columns == ["key", "amount", "code"]
            assert rows.rows() == [("r1", 10.25, "A")]
            return

        with pytest.raises(ToolError) as caught:
            _laid_out(tmp_path, text, **config)

        if outcome == "refused_config":
            assert isinstance(caught.value, ConfigError)
            assert "skip_rows" in str(caught.value)

    @pytest.mark.parametrize(
        ("text", "config", "columns", "height"),
        [
            pytest.param("key,amount\n", {}, ["key", "amount"], 0, id="a_header_and_no_rows_is_a_frame_of_no_rows"),
            pytest.param(
                "r1,10.25\n",
                {"has_header": False},
                ["column_1", "column_2"],
                1,
                id="and_a_file_that_names_nothing_gets_positions",
            ),
        ],
    )
    def test_a_file_with_no_rows_and_a_file_with_no_header_both_still_carry_columns(
        self,
        tmp_path: Path,
        text: str,
        config: Mapping[str, Any],
        columns: Sequence[str],
        height: int,
    ) -> None:
        """An output port must carry its schema even when it carries no rows, so both of these work.

        A month with nothing in it is a real delivery, and the header line is what says so; the
        frame that comes back has no rows and every column, which is what a downstream step
        compiles its expressions against.

        Positional names are the honest answer to a file that names nothing, and they are pinned by
        value because they end up in the configuration of every step downstream -- which makes them
        an interface rather than an implementation detail. They are also a warning sign: a flow
        naming ``column_2`` breaks the day somebody inserts a column, so the intended next step is
        a rename.
        """
        rows = _laid_out(tmp_path, text, **config)

        assert rows.columns == list(columns)
        assert rows.height == height
        assert rows.width == len(columns), "an output port must carry its schema even with no rows"

    @pytest.mark.parametrize(
        ("header", "named"),
        [
            pytest.param("key,amount,key", "key", id="a_heading_used_twice"),
            pytest.param("key,,code", "2", id="a_heading_left_blank"),
            pytest.param("key,  ,code", "2", id="a_heading_of_only_spaces_is_blank_too_as_in_a_workbook"),
        ],
    )
    def test_a_header_line_that_repeats_or_omits_a_name_is_refused_rather_than_renamed(
        self,
        tmp_path: Path,
        header: str,
        named: str,
    ) -> None:
        """A header line is a schema, and a schema that had to be guessed is not one.

        Left to the reader underneath, the second ``key`` arrives renamed and the blank heading
        arrives as a placeholder in that reader's own vocabulary. Both are silent, and both end the
        same way: a step downstream names ``key``, gets whichever column came first, and the run
        reports success on the wrong column. Two columns headed the same thing is a defect in the
        file, and it is cheaper to say so than to pick one.
        """
        with pytest.raises(ConfigError) as caught:
            _laid_out(tmp_path, f"{header}\nr1,10.25,A\n")

        message = str(caught.value)
        assert named in message, "which heading is the problem"
        assert "_duplicated" not in message and "UNNAMED" not in message, (
            f"the reader's own repair must not appear in a refusal about refusing to repair: {message}"
        )


# ---------------------------------------------------------------------------
# Refusals about the file itself
# ---------------------------------------------------------------------------


class TestRefusalsAboutTheFile:
    """Every way this tool can fail to read at all, in this library's words rather than a reader's."""

    @pytest.mark.parametrize(
        ("delimiter", "accepted"),
        [
            pytest.param("\t", True, id="a_tab"),
            pytest.param(";", True, id="a_semicolon"),
            pytest.param("||", False, id="two_characters"),
            pytest.param('"', False, id="the_quote_character"),
            pytest.param("\n", False, id="a_line_ending"),
        ],
    )
    def test_a_delimiter_is_one_character_and_not_the_quote_as_the_writer_requires(
        self, tmp_path: Path, delimiter: str, accepted: bool
    ) -> None:
        """Checked before the file is read, in the same words as write_csv, so both tools agree."""
        target = _file(tmp_path / "extract.csv", f"key{delimiter}code\nr1{delimiter}A\n")
        tool = ReadCsv()
        config = apply_defaults(tool.config_template(), {"path": str(target), "delimiter": delimiter})

        if accepted:
            tool.validate_config(config)
            assert _read({"path": str(target), "delimiter": delimiter}).columns == ["key", "code"]
            return
        with pytest.raises(ConfigError) as caught:
            tool.validate_config(config)
        assert "'delimiter'" in str(caught.value)

    @pytest.mark.parametrize(
        ("situation", "name"),
        [
            pytest.param("absent", "extract.csv", id="a_file_that_was_never_delivered"),
            pytest.param("directory", "extract.csv", id="a_path_that_is_a_folder_of_that_name"),
            pytest.param("empty", "extract.csv", id="a_file_with_not_a_byte_in_it"),
            pytest.param("wrong_format", "extract.xlsx", id="a_workbook_is_the_other_tool"),
            pytest.param("wrong_format", "extract", id="and_no_extension_at_all"),
        ],
    )
    def test_a_path_this_tool_cannot_read_is_refused_while_there_is_an_author_to_tell(
        self,
        tmp_path: Path,
        situation: str,
        name: str,
    ) -> None:
        """Four failures with one thing in common: the reader underneath explains none of them well.

        An absent file arrives as a bare file-not-found mentioning neither the flow nor the step, a
        folder arrives as a permission error, and a file with nothing in it arrives as an error
        recommending a keyword argument of the reader's own. So all of them are reported here,
        naming the path, as configuration faults an author can act on rather than as surprises
        inside the tool.

        The last two are the extension check, made against the configuration alone before anything
        is opened -- which is the moment there is somebody to tell. Its refusal names the format
        this tool handles, so a reader who arrived at the wrong tool of the pair is redirected
        rather than merely stopped.

        A file that is absent is refused by ``validate_inputs``, which the contract runs before
        ``execute``: that is where the question "is the world this configuration describes actually
        there?" belongs, and pinning the phase is what stops the check drifting into the first
        lines of ``execute``, where every tool in this library used to keep its own version of it.
        """
        target = tmp_path / name
        if situation == "directory":
            target.mkdir()
        elif situation == "empty":
            target.write_bytes(b"")
        elif situation == "wrong_format":
            target.write_text("key,amount\nr1,10.25\n")

        with pytest.raises(ConfigError) as caught:
            _read({"path": str(target)})

        message = str(caught.value)
        assert not isinstance(caught.value, ExecutionError), "none of these is a surprise inside the tool"
        assert not any(keyword in message for keyword in _READER_KEYWORDS), (
            f"a reader's own keyword arguments are not advice an author can take: {message}"
        )
        if situation == "wrong_format":
            assert ".csv" in message, "the refusal names the format this tool handles"
            return

        assert str(target) in message, "which file, so a delivery can be chased"
        if situation == "absent":
            tool = ReadCsv()
            filled = apply_defaults(tool.config_template(), {"path": str(target)})
            with pytest.raises(ConfigError):
                tool.validate_inputs(Ports(), filled)


# ---------------------------------------------------------------------------
# The pair, the template and the catalog
# ---------------------------------------------------------------------------


class TestReadCsvInTheCatalog:
    """The two tools agree with each other, and this one is described and resolvable by name."""

    def test_a_frame_written_by_the_other_tool_reads_back_as_itself_digits_and_all(self, tmp_path: Path) -> None:
        """The pair has to agree, and no test of either tool on its own would notice them drifting.

        Everything the writer decides -- that a header is always written, that a null is an empty
        field, how a quote inside a field is escaped, which spelling a date and a truth value get --
        is something the reader has to assume. This is the test that holds the two together.

        One entry in ``column_types`` is needed and exactly one, which is the shape of the whole
        ruling: the identifiers, the figures, the truth values and the text all come back inferred
        correctly, and the date comes back as text until the author says it is a date, because a
        file cannot say so.

        The identifier is the point of the split between this tool and the workbook pair. A cell
        holds a double, so the same number through a workbook comes back 123456789012345696 and
        nothing says so; a file holds digits, so it comes back as itself.
        """
        target = tmp_path / "extract.csv"
        frame = pl.DataFrame(
            {
                "key": ["r1", "r2"],
                "reference": [123456789012345678, 2**53 + 1],
                "amount": [10.25, -5.5],
                "on": [dt.date(2026, 1, 31), dt.date(2026, 2, 28)],
                "settled": [True, False],
                "note": ['holds, a comma and "a quote"', None],
            }
        )

        run_tool(WriteCsv(), Ports({"rows": frame}), {"path": str(target)})
        back = _read({"path": str(target), "column_types": [{"column": "on", "type": "Date"}]})

        assert_frame_equal(back, frame, check_row_order=True)

    def test_the_template_infers_by_default_and_publishes_the_names_an_override_may_use(self) -> None:
        """Six fields, one of them the override, and three deliberate absences.

        ``path`` is a plain string and emphatically not an expression: declared as one, a loader
        pre-flighting the flow would try to compile a filename as a column reference. There is no
        ``infer_schema_length``, because inference reads the whole file and a knob there would make
        a failure configurable rather than absent. There is no ``quote``, because a file quoted
        with anything but ``"`` is one almost nothing else can read. And there is no ``sheet``: a
        text file has no tabs, which is half of why this is not the workbook tool.

        ``column_types`` is a list of small objects rather than one untyped map, which is the shape
        the contract already understands: each entry's members are type-checked, required-checked
        and reported by path, and the type name is an enum the boundary itself refuses a bad value
        against. A map of name to name would be one untyped object, and nothing inside it would be
        checked at all.

        Every published type name survives a definition file, and so does every encoding: a word a
        YAML loader rewrites would arrive as something other than what the author typed, and the
        spelling of nothing would arrive as a field nobody configured.
        """
        template = ReadCsv().config_template()

        assert set(template) == {"path", "delimiter", "encoding", "has_header", "skip_rows", "column_types"}
        assert template["path"].type == "string"
        assert template["path"].required is True
        assert template["path"].non_empty is True, "an empty path is a value, and not one to act on"
        assert template["delimiter"].default == ","
        assert template["encoding"].default == "utf8"
        assert set(template["encoding"].enum or ()) >= {"utf8"}
        assert not set(template["encoding"].enum or ()) & YAML_UNSAFE_VALUES
        assert template["has_header"].default is True
        assert template["skip_rows"].type == "integer"
        assert template["skip_rows"].default == 0
        assert expression_fields(template) == (), "nothing here is a column reference"

        entry = template["column_types"].items
        assert template["column_types"].type == "array"
        assert template["column_types"].required is False, "a file whose inference is right needs no entry at all"
        assert entry is not None and entry.type == "object"
        assert set(entry.fields or {}) == {"column", "type", "format"}
        assert (entry.fields or {})["column"].required is True
        assert (entry.fields or {})["type"].required is True
        assert tuple((entry.fields or {})["type"].enum or ()) == tuple(DECLARED_TYPES)
        assert (entry.fields or {})["format"].required is False, "a date in ISO order needs no format"

        assert set(DECLARED_TYPES) == {"String", "Int64", "Float64", "Boolean", "Date", "Datetime", "Time"}
        assert not set(DECLARED_TYPES) & YAML_UNSAFE_VALUES, "a declared type has to survive a definition file"

    def test_read_csv_conforms_and_its_example_leaves_nothing_behind_it(self) -> None:
        """Conformance runs every worked example, so a documented frame is a checked frame.

        A source's example needs a file to read, which is real I/O in a check that is meant to be
        free of side effects -- so where that file lives, and whether it is still there afterwards,
        is part of the catalog's behaviour. An example reading a relative path resolves against
        wherever the check was run from; one reading from the repository ties the catalog to this
        repository's contents; and one that leaves its file behind makes the second run of the same
        check a different check from the first. The example declares the file through ``setup``
        rather than having ``help()`` create it, because the catalog page generator calls ``help()``
        on every tool and a description must not change the process doing the describing.

        Checked twice on purpose: a documented source that could only be checked once would be a
        documented source nobody could re-check.

        Every example claims row order, which is the deliberate difference from the catalog's other
        source. A database returns rows in no order unless asked, so no example of that tool may
        claim one; a file is a sequence of lines, so this one can and should.
        """
        tool = ReadCsv()
        before = sorted(Path.cwd().iterdir())

        check_tool(tool)
        check_tool(tool)

        assert sorted(Path.cwd().iterdir()) == before, "the catalog check must not litter the working directory"
        assert tool.name == "read_csv"
        assert tool.category == "input"
        assert tool.category in CATEGORIES
        assert isinstance(tool, Tool), "validate_inputs is a required member even with no inputs"
        assert tuple(tool.input_ports) == (), "a source reads no ports; the contract constrains only outputs"
        assert tuple(tool.output_ports) == ("rows",)

        help_ = tool.help()
        assert help_.examples, "a source with no runnable example is a source nobody can copy"
        for example in help_.examples:
            assert example.setup is not None, "an example that needs a file declares it rather than writing one"
            assert example.check_row_order, "a file has an order, and a reader of one may promise it"
            configured = str(example.configuration["path"])
            assert configured.startswith("{scratch}/"), (
                f"a reader has to be able to copy this straight off the page: {configured}"
            )

        documented = " ".join((help_.detailed_description, *help_.notes)).lower()
        assert "infer" in documented, "that a column is inferred has to be documented"
        assert "null" in documented or "nothing" in documented, "and so does what a blank field becomes"

        assert "read_csv" in REGISTRY
        assert isinstance(REGISTRY.get("read_csv"), ReadCsv)
