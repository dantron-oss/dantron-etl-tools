"""Tests for the CastColumns tool: a column's type stated by the flow rather than inferred.

A type comes from wherever the data came from, and sometimes that is not the type the data *is*: a
CSV has none, a database column defined years ago holds digits as text, and two branches about to be
stacked disagree because one of them happened to see only whole numbers. The fix is always the same
-- somebody who knows says what the column is -- and this is where they say it.

It is a step rather than a field on Union, ReadTable and ReadExcel, which is the point: one concept
instead of the same words written down four times, and a type change that appears on the canvas as a
node with a reason beside it rather than as a line inside a step that looks like it is doing
something else.

The two rulings worth protecting are both about refusing to be helpful. A value that will not
convert stops the run, because a null in its place leaves a frame of the right shape with the value
silently gone. And a cast to the type a column already has is refused, like every other inert
declaration in this library.

Vocabulary is generic: the library carries no caller's nouns.
"""

from datetime import date

import polars as pl
import pytest

from dantron_etl.tool import (
    CATEGORIES,
    REGISTRY,
    ConfigError,
    DataError,
    Ports,
    check_tool,
    run_tool,
)
from dantron_etl.tools.cast_columns import CastColumns


def _rows() -> pl.DataFrame:
    """Everything arriving as text, which is what a source with no types gives."""
    return pl.DataFrame(
        {
            "record_id": ["1", "2", "3"],
            "code": ["A1", "B2", "C3"],
            "amount": ["10.5", "-3", None],
            "opened_on": ["2026-01-01", "2026-02-01", "2026-03-01"],
        }
    )


def _cast(frame: pl.DataFrame, *pairs: tuple[str, str]) -> pl.DataFrame:
    """One run through the boundary, so the envelope is enforced on every case here."""
    config = {"columns": [{"column": column, "type": declared} for column, declared in pairs]}
    return run_tool(CastColumns(), Ports({"rows": frame}), config)["rows"]


class TestReadingAColumnAsItsType:
    """Only the type changes: same rows, same order, same positions."""

    def test_a_text_column_of_digits_becomes_numbers(self) -> None:
        rows = _cast(_rows(), ("amount", "Float64"))

        assert rows.schema["amount"] == pl.Float64
        assert rows["amount"].to_list() == [10.5, -3.0, None], "a null was never a value that failed to convert"

    def test_the_column_keeps_its_position_and_the_rows_keep_their_order(self) -> None:
        """So a cast can go anywhere in a flow without moving anything a later step reads by position."""
        rows = _cast(_rows(), ("amount", "Float64"))

        assert rows.columns == _rows().columns
        assert rows["record_id"].to_list() == ["1", "2", "3"]
        assert rows.height == 3

    def test_several_columns_are_settled_in_one_step(self) -> None:
        rows = _cast(_rows(), ("record_id", "Int64"), ("opened_on", "Date"))

        assert rows.schema["record_id"] == pl.Int64
        assert rows.schema["opened_on"] == pl.Date
        assert rows["opened_on"].to_list() == [date(2026, 1, 1), date(2026, 2, 1), date(2026, 3, 1)]

    def test_an_untouched_column_is_left_exactly_as_it_was(self) -> None:
        rows = _cast(_rows(), ("amount", "Float64"))

        assert rows.drop("amount").schema == _rows().drop("amount").schema

    def test_a_frame_with_no_rows_still_changes_its_schema(self) -> None:
        """A cast is a statement about the column, not about the rows that happen to be in it."""
        rows = _cast(_rows().head(0), ("amount", "Float64"))

        assert rows.height == 0
        assert rows.schema["amount"] == pl.Float64

    def test_cast_columns_leaves_its_input_untouched(self) -> None:
        frame = _rows()
        before = frame.clone()

        _cast(frame, ("amount", "Float64"))

        assert frame.equals(before)


class TestRefusals:
    """Each names the column, because that is the fact an author needs to act."""

    def test_a_value_that_will_not_convert_stops_the_run(self) -> None:
        """A null in its place would leave a frame of the right shape with the value gone."""
        with pytest.raises(DataError) as caught:
            _cast(_rows(), ("code", "Int64"))

        assert "code" in str(caught.value)
        assert "Int64" in str(caught.value), "which type was asked for is half of what a reader needs"

    def test_a_refusal_about_values_reports_the_offending_rows(self) -> None:
        """So the answer is in the message rather than in a query the reader has to write first."""
        with pytest.raises(DataError) as caught:
            _cast(_rows(), ("code", "Int64"))

        assert caught.value.offending_rows == 3
        assert caught.value.total_rows == 3
        assert caught.value.examples, "a count without an example sends the reader back to the data"

    def test_a_pair_the_engine_refuses_outright_is_still_a_refusal_and_not_a_crash(self) -> None:
        """Text does not become a truth value here, and the report of that must not itself throw."""
        with pytest.raises(DataError) as caught:
            _cast(_rows(), ("code", "Boolean"))

        assert "code" in str(caught.value)

    def test_casting_a_column_to_the_type_it_already_has_is_refused(self) -> None:
        with pytest.raises(ConfigError) as caught:
            _cast(_rows(), ("code", "String"))

        assert "code" in str(caught.value)

    def test_a_column_the_frame_does_not_carry_is_refused_naming_the_columns_it_has(self) -> None:
        with pytest.raises(ConfigError) as caught:
            _cast(_rows(), ("quantity", "Int64"))

        message = str(caught.value)
        assert "quantity" in message
        assert "record_id" in message, "listing what is present is what turns a complaint into an answer"

    def test_the_same_column_settled_twice_is_refused(self) -> None:
        """Two answers for one column is a configuration edited halfway, not a preference."""
        with pytest.raises(ConfigError) as caught:
            _cast(_rows(), ("amount", "Float64"), ("amount", "String"))

        assert "amount" in str(caught.value)

    def test_an_empty_list_is_refused(self) -> None:
        """`required` alone accepts an empty list, which is a step that does nothing."""
        with pytest.raises(ConfigError):
            run_tool(CastColumns(), Ports({"rows": _rows()}), {"columns": []})

    @pytest.mark.parametrize("height", [0, 3])
    def test_a_schema_refusal_does_not_wait_for_rows(self, height: int) -> None:
        with pytest.raises(ConfigError):
            _cast(_rows().head(height), ("quantity", "Int64"))


class TestCastColumnsInTheCatalog:
    """Declared, described, and documented with examples that run."""

    def test_cast_columns_conforms_and_its_examples_change_a_type(self) -> None:
        tool = CastColumns()
        check_tool(tool)

        assert tool.name == "cast_columns"
        assert tool.category in CATEGORIES
        assert tool.input_ports == ("rows",)
        assert tool.output_ports == ("rows",)

        for example in tool.help().examples:
            given, produced = example.inputs["rows"], example.outputs["rows"]
            assert produced.columns == given.columns, "a cast moves nothing"
            assert produced.height == given.height
            assert produced.schema != given.schema, "an example has to actually change a type"

    def test_cast_columns_is_in_the_shipped_catalog(self) -> None:
        assert REGISTRY.get("cast_columns") is not None

    def test_it_names_the_same_types_the_csv_reader_does(self) -> None:
        """One vocabulary, so a reader who knows one tool's spellings knows the other's."""
        from dantron_etl.column_types import DECLARED_TYPES
        from dantron_etl.tools.read_csv import DECLARED_TYPES as CSV_TYPES

        declared = (CastColumns().config_template()["columns"].items.fields or {})["type"]
        assert tuple(declared.enum or ()) == tuple(DECLARED_TYPES)
        assert tuple(CSV_TYPES) == tuple(DECLARED_TYPES), "two spellings of one vocabulary is how they drift"


class TestReportingWhatCouldNotBeRead:
    """`on_invalid: empty` is the other honest answer, and it has to say how big the problem was.

    Three unreadable values out of forty-six thousand should not stop a nightly build, and a flow
    that says so gets nulls in their place. What makes that safe rather than silent is the count: how
    many, out of how many, which raises the step to a warning so the run carries on and the size of
    the problem carries on with it.

    The count is taken before the values are replaced, and that ordering is the whole trick -- a null
    that failed to convert and a null that was always there are indistinguishable afterwards.
    """

    def test_unreadable_values_become_empty_and_the_rest_convert(self) -> None:
        frame = pl.DataFrame({"amount": ["10.5", "oops", "-3", None]})
        rows = run_tool(
            CastColumns(),
            Ports({"rows": frame}),
            {"columns": [{"column": "amount", "type": "Float64"}], "on_invalid": "empty"},
        )["rows"]

        assert rows["amount"].to_list() == [10.5, None, -3.0, None]
        assert rows.schema["amount"] == pl.Float64

    def test_the_note_says_how_many_out_of_how_many(self) -> None:
        """A warning without a size is a warning nobody can act on."""
        frame = pl.DataFrame({"amount": ["10.5", "oops", "-3", "also bad"]})
        result = run_tool(
            CastColumns(),
            Ports({"rows": frame}),
            {"columns": [{"column": "amount", "type": "Float64"}], "on_invalid": "empty"},
        )

        assert len(result.notes) == 1
        note = result.notes[0]
        assert "2 of 4" in note
        assert "amount" in note
        assert "Float64" in note

    def test_a_null_that_was_always_null_is_not_counted_as_a_loss(self) -> None:
        """Counting it would report a problem that is not there, on every column that allows nulls."""
        frame = pl.DataFrame({"amount": ["10.5", None, None]})
        result = run_tool(
            CastColumns(),
            Ports({"rows": frame}),
            {"columns": [{"column": "amount", "type": "Float64"}], "on_invalid": "empty"},
        )

        assert result.notes == (), "nothing failed to convert, so there is nothing to report"

    def test_nothing_unreadable_reports_nothing(self) -> None:
        """A clean cast is not a warning, or every flow would be one."""
        result = run_tool(
            CastColumns(),
            Ports({"rows": pl.DataFrame({"amount": ["1", "2"]})}),
            {"columns": [{"column": "amount", "type": "Float64"}], "on_invalid": "empty"},
        )

        assert result.notes == ()

    def test_error_is_the_default_so_silence_is_never_what_you_get_by_not_choosing(self) -> None:
        """The dangerous mode is the one you have to ask for, which is the right way round."""
        frame = pl.DataFrame({"amount": ["10.5", "oops"]})

        with pytest.raises(DataError):
            run_tool(CastColumns(), Ports({"rows": frame}), {"columns": [{"column": "amount", "type": "Float64"}]})

    def test_each_column_reports_separately(self) -> None:
        """So a reader can see which column is the problem rather than a total across all of them."""
        frame = pl.DataFrame({"amount": ["1", "bad"], "count": ["2", "worse"]})
        result = run_tool(
            CastColumns(),
            Ports({"rows": frame}),
            {
                "columns": [{"column": "amount", "type": "Float64"}, {"column": "count", "type": "Int64"}],
                "on_invalid": "empty",
            },
        )

        assert len(result.notes) == 2
        assert any("amount" in note for note in result.notes)
        assert any("count" in note for note in result.notes)

    def test_on_invalid_offers_no_word_a_definition_file_would_rewrite(self) -> None:
        """'empty' rather than 'null': an unquoted null in YAML IS the null value."""
        from dantron_etl.tool import YAML_UNSAFE_VALUES

        choices = set(CastColumns().config_template()["on_invalid"].enum or ())
        assert choices == {"error", "empty"}
        assert not choices & YAML_UNSAFE_VALUES
