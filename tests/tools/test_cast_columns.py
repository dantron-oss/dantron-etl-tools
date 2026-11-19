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

from datetime import date, datetime
from typing import Any
from decimal import Decimal

import polars as pl
import pytest

from dantron_etl.authoring import CATEGORIES, ConfigError, DataError, Ports, REGISTRY, check_tool, run_tool
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
    """`on_invalid: set_null` is the other honest answer, and it has to say how big the problem was.

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
            {"columns": [{"column": "amount", "type": "Float64"}], "on_invalid": "set_null"},
        )["rows"]

        assert rows["amount"].to_list() == [10.5, None, -3.0, None]
        assert rows.schema["amount"] == pl.Float64

    def test_the_note_says_how_many_out_of_how_many(self) -> None:
        """A warning without a size is a warning nobody can act on."""
        frame = pl.DataFrame({"amount": ["10.5", "oops", "-3", "also bad"]})
        result = run_tool(
            CastColumns(),
            Ports({"rows": frame}),
            {"columns": [{"column": "amount", "type": "Float64"}], "on_invalid": "set_null"},
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
            {"columns": [{"column": "amount", "type": "Float64"}], "on_invalid": "set_null"},
        )

        assert result.notes == (), "nothing failed to convert, so there is nothing to report"

    def test_nothing_unreadable_reports_nothing(self) -> None:
        """A clean cast is not a warning, or every flow would be one."""
        result = run_tool(
            CastColumns(),
            Ports({"rows": pl.DataFrame({"amount": ["1", "2"]})}),
            {"columns": [{"column": "amount", "type": "Float64"}], "on_invalid": "set_null"},
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
                "on_invalid": "set_null",
            },
        )

        assert len(result.notes) == 2
        assert any("amount" in note for note in result.notes)
        assert any("count" in note for note in result.notes)

    def test_on_invalid_offers_no_word_a_definition_file_would_rewrite(self) -> None:
        """'set_null' rather than 'null': an unquoted null in YAML IS the null value."""
        from dantron_etl.authoring import YAML_UNSAFE_VALUES

        choices = set(CastColumns().config_template()["on_invalid"].enum or ())
        assert choices == {"error", "set_null"}
        assert not choices & YAML_UNSAFE_VALUES


class TestADecimalBecomesAWholeNumberOnlyWhenItIsOne:
    """A decimal read as Int64 refuses unless every value is already whole (the owner's ruling).

    The engine cuts the fraction off without a word: 1.99 becomes 1 and -2.7 becomes -2. So a value
    with a fraction is a value that does not read as a whole number. Under 'error' it stops the run,
    naming the column, some of the values, and saying to round first. Under 'set_null' it is
    emptied, counted and marked, like any other value that does not read.
    """

    def test_a_fraction_stops_the_run_naming_the_column_the_values_and_round(self) -> None:
        frame = pl.DataFrame({"key": ["a", "b", "c", "d"], "seats": [12.0, 1.99, -2.7, None]})

        with pytest.raises(DataError) as caught:
            run_tool(CastColumns(), Ports({"rows": frame}), {"columns": [{"column": "seats", "type": "Int64"}]})

        message = str(caught.value)
        for word in ("seats", "1.99", "-2.7", "Int64", "round"):
            assert word in message, message
        assert caught.value.offending_rows == 2

    def test_whole_values_held_as_decimals_still_read(self) -> None:
        frame = pl.DataFrame({"seats": [12.0, -3.0, None, 0.0]})
        assert _cast(frame, ("seats", "Int64"))["seats"].to_list() == [12, -3, None, 0]

    def test_an_exact_decimal_type_is_held_to_the_same_rule(self) -> None:
        whole = pl.DataFrame({"seats": pl.Series([Decimal("12.00"), Decimal("3.00")], dtype=pl.Decimal(10, 2))})
        assert _cast(whole, ("seats", "Int64"))["seats"].to_list() == [12, 3]

        cut = pl.DataFrame({"seats": pl.Series([Decimal("12.50")], dtype=pl.Decimal(10, 2))})
        with pytest.raises(DataError):
            _cast(cut, ("seats", "Int64"))

    def test_set_null_empties_counts_and_marks_a_fraction_like_any_value_that_did_not_read(self) -> None:
        frame = pl.DataFrame({"seats": [12.0, 1.99, None]})
        config = {"columns": [{"column": "seats", "type": "Int64"}], "on_invalid": "set_null"}

        result = run_tool(CastColumns(), Ports({"rows": frame}), config)

        assert result["rows"]["seats"].to_list() == [12, None, None]
        assert any("1 of 3" in note and "1.99" in note for note in result.notes), result.notes

    def test_a_decimal_read_as_another_decimal_type_or_as_text_is_not_held_to_it(self) -> None:
        frame = pl.DataFrame({"seats": [1.99]})
        assert _cast(frame, ("seats", "String"))["seats"].to_list() == ["1.99"]

    def test_text_digits_and_whole_numbers_read_as_before(self) -> None:
        frame = pl.DataFrame({"a": ["12", "-3"], "b": pl.Series([1, 2], dtype=pl.Int32)})
        result = _cast(frame, ("a", "Int64"), ("b", "Int64"))
        assert result["a"].to_list() == [12, -3] and result["b"].to_list() == [1, 2]


def _late_in_toronto() -> pl.DataFrame:
    """23:30 on 31 January in Toronto, which is 04:30 on 1 February in UTC, and a zone-less twin."""
    moment = datetime(2026, 2, 1, 4, 30)
    return pl.DataFrame({"paid_at": [moment, None], "noted_at": [moment, None]}).with_columns(
        pl.col("paid_at").dt.replace_time_zone("UTC").dt.convert_time_zone("America/Toronto")
    )


class TestATimestampInAZoneIsReadAsADateInAStatedZone:
    """Which day a timestamp falls on depends on the zone, so the author states it (the owner's ruling).

    No default: a cast to Date of a zone-aware timestamp without 'in_zone' is refused by name, on
    the schema, before any row is read. 'in_zone: UTC' gives the UTC date, which is what the cast
    gave before. 'in_zone' anywhere it would do nothing is refused too.
    """

    def _cast_paid(self, **entry: Any) -> pl.DataFrame:
        config = {"columns": [{"column": "paid_at", "type": "Date", **entry}]}
        return run_tool(CastColumns(), Ports({"rows": _late_in_toronto()}), config)["rows"]

    @pytest.mark.parametrize("height", [2, 0])
    def test_without_a_zone_it_is_refused_naming_the_column_its_zone_and_the_setting(self, height: int) -> None:
        config = {"columns": [{"column": "paid_at", "type": "Date"}]}
        with pytest.raises(ConfigError) as caught:
            run_tool(CastColumns(), Ports({"rows": _late_in_toronto().head(height)}), config)
        message = str(caught.value)
        for word in ("paid_at", "America/Toronto", "in_zone", "UTC"):
            assert word in message, message

    def test_utc_gives_the_date_the_cast_gave_before(self) -> None:
        assert self._cast_paid(in_zone="UTC")["paid_at"].to_list() == [date(2026, 2, 1), None]

    def test_the_stated_zone_decides_the_day(self) -> None:
        assert self._cast_paid(in_zone="America/Toronto")["paid_at"].to_list() == [date(2026, 1, 31), None]

    def test_a_timestamp_with_no_zone_needs_none_and_refuses_one(self) -> None:
        plain = {"columns": [{"column": "noted_at", "type": "Date"}]}
        assert run_tool(CastColumns(), Ports({"rows": _late_in_toronto()}), plain)["rows"]["noted_at"].to_list() == [
            date(2026, 2, 1),
            None,
        ]
        with pytest.raises(ConfigError) as caught:
            run_tool(
                CastColumns(),
                Ports({"rows": _late_in_toronto()}),
                {"columns": [{"column": "noted_at", "type": "Date", "in_zone": "UTC"}]},
            )
        assert "in_zone" in str(caught.value) and "noted_at" in str(caught.value)

    def test_a_zone_beside_a_type_other_than_date_is_refused(self) -> None:
        with pytest.raises(ConfigError) as caught:
            run_tool(
                CastColumns(),
                Ports({"rows": _late_in_toronto()}),
                {"columns": [{"column": "paid_at", "type": "String", "in_zone": "UTC"}]},
            )
        assert "in_zone" in str(caught.value)

    def test_a_zone_nobody_has_heard_of_is_refused_before_any_frame(self) -> None:
        with pytest.raises(ConfigError) as caught:
            CastColumns().validate_config(
                {"columns": [{"column": "paid_at", "type": "Date", "in_zone": "Mars/Olympus"}], "on_invalid": "error"}
            )
        assert "Mars/Olympus" in str(caught.value)
