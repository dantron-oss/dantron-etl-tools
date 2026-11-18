"""Tests for the RecordId tool: a number per row, following the order the rows arrived in.

RecordId takes one input port, ``rows``, and produces one, ``rows``. Everything that came in comes
back, unmoved, with one more column on the end: a whole number that starts at ``start`` and counts
up by one per row.

Six decisions are pinned here, because each has a quiet failure mode.

**The number follows arrival order, and the rows do not move.** RecordId sorts nothing and decides
nothing. That is what makes it only meaningful after an explicit sort step, and it is why the test
below feeds it a frame that is in no useful order and checks that the frame comes back in exactly
that order.

**``start: 0`` is honoured, and so is a negative start.** Zero is a value, not an absence, but it is
also falsy, and the contract's own default checking skips a default that reads as nothing. A number
scheme that silently began at 1 when its author asked for 0 would be off by one everywhere, and
nothing would say so.

**The new column goes last, and every column already there survives with its type.** A step that
rebuilt the frame to make room would show up as a changed type or a changed column order.

**The column is a 64-bit integer, declared rather than inferred.** On a frame with no rows there is
nothing to infer a type from, and a column that came back typed as nothing at all would break the
next join -- on the day the upstream filter happened to match nothing, and not before.

**A numbering that will not fit in a 64-bit integer is refused, not attempted.** ``start`` is a
Python integer and has no ceiling. The column it goes into does. A start past the ceiling, or a
start that runs past it part way down the frame, has to be refused by name. Left alone it produces
numbers that are not the numbers the file asked for, and reports success.

**Refusals are read off the schema.** Each refusal case runs twice, once on a populated frame and
once on a frame with no rows, which is what shows the refusal came from the column names and the
configuration rather than from the values. Writing over a column that is already there is refused
through the contract's own shared check, so the wording is the same one every other column-writing
tool in the catalog uses.

Vocabulary is deliberately generic -- zone, amount, code -- because the library carries no caller's
nouns.
"""

from collections.abc import Mapping
from datetime import date
from typing import Any, Final

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.tool import (
    CATEGORIES,
    REGISTRY,
    ConfigError,
    Ports,
    check_tool,
    expression_fields,
    run_tool,
)
from dantron_etl.tools.record_id import RecordId


def _rows() -> pl.DataFrame:
    """A frame in no useful order, carrying four types and a missing value in each nullable one.

    Deliberately not sorted by anything: ``code`` runs backwards, ``amount`` jumps around, and the
    zones repeat. If the numbering followed any value in the frame rather than the order the rows
    arrived in, it would show up against this frame and not against a tidy one.
    """
    return pl.DataFrame(
        {
            "code": ["d", "a", "c", "b"],
            "zone": ["north", None, "north", "east"],
            "amount": [10.0, -5.0, None, 2.5],
            "opened": [date(2024, 3, 1), date(2024, 1, 1), None, date(2024, 2, 1)],
        }
    )


class TestRecordIdNumbering:
    """What the new column holds, and that nothing else about the frame changes."""

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            pytest.param({"output": "row_no"}, [1, 2, 3, 4], id="the_numbering_starts_at_one_by_default"),
            pytest.param({"output": "row_no", "start": 1}, [1, 2, 3, 4], id="one_may_also_be_said_out_loud"),
            pytest.param({"output": "row_no", "start": 0}, [0, 1, 2, 3], id="a_start_of_zero_is_honoured"),
            pytest.param({"output": "row_no", "start": 100}, [100, 101, 102, 103], id="a_start_part_way_up"),
            pytest.param(
                {"output": "row_no", "start": -2},
                [-2, -1, 0, 1],
                id="a_negative_start_counts_up_through_zero",
            ),
        ],
    )
    def test_the_column_counts_up_by_one_from_the_start_that_was_asked_for(
        self,
        config: Mapping[str, Any],
        expected: list[int],
    ) -> None:
        """``start: 0`` is the case that matters most, because zero reads as an absence.

        The contract fills a field nobody configured with its declared default, and it type-checks a
        declared default only when that default is not nothing. A zero travelling through that path
        as if it were nothing would produce a numbering that began at 1 while the configuration on
        the page said 0, and every downstream number would be one out with nothing to show why.
        """
        frame = _rows()

        result = run_tool(RecordId(), Ports({"rows": frame}), config)

        assert result.names == ("rows",), "one input, one output, both named for what they carry"
        assert result["rows"]["row_no"].to_list() == expected
        assert result["rows"].schema["row_no"] == pl.Int64, "a row number is a 64-bit integer, always"

    def test_the_rows_do_not_move_and_every_input_column_survives_with_its_type(self) -> None:
        """The numbering follows arrival order, which is why a sort has to come first to mean anything.

        Comparing the output with the new column dropped against the input pins the row order, the
        column order, the types and the values in one assertion. The new column is then checked to
        be the last one, because a number wedged into the middle of a frame moves every column after
        it and breaks anything reading by position.
        """
        frame = _rows()

        result = run_tool(RecordId(), Ports({"rows": frame}), {"output": "row_no"})

        produced = result["rows"]
        assert produced.columns == [*frame.columns, "row_no"], "the new column goes on the end"
        assert_frame_equal(produced.drop("row_no"), frame)
        assert produced["code"].to_list() == ["d", "a", "c", "b"], "arrival order, not any order of values"
        assert produced.height == frame.height, "numbering rows adds no rows and removes none"

    def test_a_frame_with_no_rows_comes_back_with_the_column_present_and_typed(self) -> None:
        """The type is declared, never inferred, because an empty frame has nothing to infer from.

        A column that came back typed as nothing at all would satisfy every check a reader is likely
        to write and then fail at the next join -- on the day the step upstream matched no rows, and
        not before.
        """
        empty = _rows().head(0)

        result = run_tool(RecordId(), Ports({"rows": empty}), {"output": "row_no", "start": 7})

        produced = result["rows"]
        assert produced.height == 0
        assert produced.columns == [*empty.columns, "row_no"]
        assert produced.schema["row_no"] == pl.Int64, "not a column of nothing; the next step reads this type"
        assert produced.schema == {**empty.schema, "row_no": pl.Int64}

    def test_a_single_row_gets_the_start_and_nothing_else(self) -> None:
        """The smallest frame that can carry a number, where an off-by-one has nowhere to hide."""
        frame = _rows().head(1)

        result = run_tool(RecordId(), Ports({"rows": frame}), {"output": "row_no", "start": 5})

        assert result["rows"]["row_no"].to_list() == [5]

    def test_record_id_leaves_its_input_untouched(self) -> None:
        """The boundary compares input schemas; the frame itself must be unchanged too.

        A tool that added its column in place would pass a boundary check that only compares schemas
        of a frame it had already replaced, and would leak its column into the other branch of any
        fan-out.
        """
        frame = _rows()
        before = frame.clone()

        run_tool(RecordId(), Ports({"rows": frame}), {"output": "row_no"})

        assert_frame_equal(frame, before)


def _refusal_cases() -> list[Any]:
    """(config, the text its refusal must quote) for each way a numbering can be wrong."""
    return [
        pytest.param({"output": "code"}, "code", id="an_output_name_a_column_already_uses"),
        pytest.param({"output": "amount"}, "amount", id="an_output_name_a_typed_column_already_uses"),
        pytest.param({"output": ""}, "output", id="an_output_name_that_is_empty"),
        pytest.param({"output": "   "}, "output", id="an_output_name_that_is_only_whitespace"),
        pytest.param({"output": None}, "output", id="an_output_written_with_no_value"),
        pytest.param({}, "output", id="no_output_at_all"),
        pytest.param({"output": "row_no", "start": 2.5}, "start", id="a_start_that_is_not_a_whole_number"),
        pytest.param({"output": "row_no", "start": "1"}, "start", id="a_start_written_as_text"),
        pytest.param({"output": "row_no", "start": True}, "start", id="a_start_that_is_a_truth_value"),
        pytest.param({"output": "row_no", "start": None}, "start", id="a_start_written_with_no_value"),
        pytest.param({"output": 7}, "output", id="an_output_name_that_is_not_text"),
        pytest.param({"output": "row_no", "first": 1}, "first", id="a_field_nobody_declared"),
    ]


class TestRecordIdRefusals:
    """What RecordId refuses, and in whose words: this library's own family, before any row is read."""

    @pytest.mark.parametrize(("config", "named"), _refusal_cases())
    @pytest.mark.parametrize("height", [4, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_numbering_the_schema_cannot_carry_is_refused_by_name(
        self,
        config: Mapping[str, Any],
        named: str,
        height: int,
    ) -> None:
        """Taking a name already in use is the hazard, and it has to be refused in shared words.

        Whoever writes this step does not know every column of the frame they are writing into, so a
        collision taken silently destroys a column and reports success. The contract already owns
        that decision for every column-writing tool in the catalog; this tool has to use it rather
        than invent a second way of deciding the same thing, or the catalog grows two wordings for
        one rule.

        A start written with no value is the other case worth reading twice. A definition file
        spells that ``start:``, which arrives here as nothing at all -- and a tool that took it as
        "not configured" would number from 1 under a scheme its author never chose.
        """
        frame = _rows().head(height)

        with pytest.raises(ConfigError) as caught:
            run_tool(RecordId(), Ports({"rows": frame}), config)

        message = str(caught.value)
        assert named in message, f"refusal should name {named!r}: {message}"

    def test_an_output_name_already_in_use_is_refused_in_the_shared_wording(self) -> None:
        """There is no replace option here, so the shared check is asked with nothing to replace.

        The wording the shared check produces is what makes the fix one keystroke: it names the
        field, names the column, and says what to do instead. Pinning a phrase from it is what stops
        this tool growing a private message that says the same thing differently.
        """
        frame = _rows()

        with pytest.raises(ConfigError) as caught:
            run_tool(RecordId(), Ports({"rows": frame}), {"output": "zone"})

        message = str(caught.value)
        assert "output" in message
        assert "zone" in message
        assert "already carries" in message, (
            "the shared check owns this refusal; a private wording here would be a second way of "
            f"deciding one rule: {message}"
        )

    def test_the_template_declares_a_required_output_and_a_whole_number_start(self) -> None:
        """``non_empty`` is the half ``required`` cannot do, and the default is a real number.

        ``required`` asks whether a value arrived; an empty string arrived. ``non_empty`` is what
        refuses it, so the refusal names the field rather than surfacing later as a frame with a
        column called nothing.

        ``start`` is declared an integer rather than a number, so a fractional start is refused by
        the template instead of silently truncating somewhere inside the engine.
        """
        template = RecordId().config_template()

        assert set(template) == {"output", "start"}, "two fields; a row number has nothing else to say"
        assert template["output"].type == "string"
        assert template["output"].required is True
        assert template["output"].non_empty is True, "an empty name is a value, so 'required' alone accepts it"
        assert template["start"].type == "integer", "a row number counts; it does not measure"
        assert template["start"].required is False
        assert template["start"].default == 1, "the common case needs no configuration"
        assert expression_fields(template) == (), "a column name is not a computed value; there is no seam here"


class TestRecordIdInTheCatalog:
    """Described, documented with examples that actually run, and resolvable by name."""

    def test_record_id_conforms_and_an_example_shows_the_numbering_it_produces(self) -> None:
        """Conformance runs every worked example, so a documented numbering is a checked numbering.

        Row order is the thing this tool promises not to change, so its examples ask for the ordered
        comparison. Without that, an example would pass while the rows came back in another order
        carrying the same numbers.
        """
        tool = RecordId()

        check_tool(tool)

        assert tool.name == "record_id"
        assert tool.category == "preparation", "changing the shape of a frame without deciding anything"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",)

        examples = tool.help().examples
        assert examples, "a tool nobody can copy from is a tool nobody uses"
        assert all(example.check_row_order for example in examples), (
            "this tool's promise is that the rows do not move, so its examples must be compared in order"
        )
        for example in examples:
            produced = example.outputs["rows"]
            assert produced.columns[-1] == example.configuration["output"], (
                "the documented output has to show the new column on the end, which is where it goes"
            )
            assert produced.schema[example.configuration["output"]] == pl.Int64

    def test_record_id_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in tests/core/test_registry.py."""
        assert "record_id" in REGISTRY
        assert isinstance(REGISTRY.get("record_id"), RecordId)


_INT64_MAX: Final[int] = 2**63 - 1
_INT64_MIN: Final[int] = -(2**63)


class TestRecordIdStaysInsideASixtyFourBitInteger:
    """A numbering that cannot fit in the column it is declared to live in is refused.

    ``start`` is declared an integer, and Python integers have no ceiling. The column the numbers go
    into is a signed 64-bit one. So a start can be written that the column cannot hold, and the
    numbers it produces are not the numbers the file asked for.

    There are two shapes of that, and neither announces itself.

    A start that fits on its own, but runs off the end part way down the frame. The first rows get
    the numbers they were promised. Then the count passes the ceiling and comes back round at the
    bottom, so the column stops counting up and starts counting up from a large negative number. The
    tool reports success. An id column that is supposed to be unique and increasing is neither, and
    the only way to find out is to read the numbers.

    A start that does not fit at all. That is a perfectly legal number to write in a definition
    file, and it can only fail inside the engine. Left to the engine it arrives as an execution
    error naming a phase, not as a refusal naming the field. This whole class of tool promises one
    error family and one field name, and a number nobody can hold is a configuration problem.

    The rule being pinned: refuse when ``start`` does not fit, and refuse when the last number the
    frame would be given -- ``start + height - 1`` -- does not fit. The second one needs the frame,
    so it can only be answered where the frame is.
    """

    @pytest.mark.parametrize(
        ("start", "height"),
        [
            pytest.param(_INT64_MAX, 4, id="a_start_at_the_ceiling_with_rows_left_to_number"),
            # Four rows from here land on MAX-2, MAX-1, MAX and one past it. The frame is what
            # breaks this start, not the start itself, so it has to be refused on the rows.
            pytest.param(_INT64_MAX - 2, 4, id="a_start_that_runs_one_row_past_the_ceiling"),
            pytest.param(_INT64_MAX + 1, 4, id="a_start_one_past_the_ceiling"),
            pytest.param(_INT64_MAX + 1, 0, id="a_start_past_the_ceiling_with_no_rows_to_number"),
            pytest.param(_INT64_MIN - 1, 4, id="a_start_one_below_the_floor"),
            pytest.param(_INT64_MIN - 1, 0, id="a_start_below_the_floor_with_no_rows_to_number"),
            pytest.param(2**70, 4, id="a_start_far_past_anything_the_column_can_hold"),
        ],
    )
    def test_a_numbering_that_will_not_fit_is_refused_by_name(self, start: int, height: int) -> None:
        """Every one of these has to arrive as a ConfigError naming ``start``.

        Today several of them do something else entirely. They produce a column of the wrong type,
        or a column of nothing, or a column that wraps round to negative numbers, and they report
        success either way. All three are worse than a refusal, because nothing downstream can tell
        that anything went wrong.
        """
        frame = _rows().head(height)

        with pytest.raises(ConfigError) as caught:
            run_tool(RecordId(), Ports({"rows": frame}), {"output": "row_no", "start": start})

        assert "start" in str(caught.value), f"refusal should name the field to fix: {caught.value}"

    def test_a_start_that_fits_but_overflows_part_way_down_says_which_number_broke(self) -> None:
        """The start is legal on its own, so the refusal has to explain what the frame did to it.

        Someone reading "start must fit in a 64-bit integer" next to a start that plainly does fit
        has been told nothing. Naming the number the last row would have been given is what makes
        the refusal actionable.
        """
        frame = _rows()
        last_number = _INT64_MAX + frame.height - 1

        with pytest.raises(ConfigError) as caught:
            run_tool(RecordId(), Ports({"rows": frame}), {"output": "row_no", "start": _INT64_MAX})

        message = str(caught.value)
        assert "start" in message, f"refusal should name the field to fix: {message}"
        assert str(last_number) in message, (
            f"a start that fits on its own needs the number that did not fit spelled out: {message}"
        )

    @pytest.mark.parametrize(
        ("start", "height", "expected"),
        [
            pytest.param(_INT64_MAX, 1, [_INT64_MAX], id="one_row_numbered_exactly_at_the_ceiling"),
            pytest.param(
                _INT64_MAX - 3,
                4,
                [_INT64_MAX - 3, _INT64_MAX - 2, _INT64_MAX - 1, _INT64_MAX],
                id="a_frame_that_ends_exactly_on_the_ceiling",
            ),
            pytest.param(
                _INT64_MIN,
                4,
                [_INT64_MIN, _INT64_MIN + 1, _INT64_MIN + 2, _INT64_MIN + 3],
                id="a_start_exactly_on_the_floor",
            ),
            pytest.param(_INT64_MAX, 0, [], id="a_start_at_the_ceiling_with_no_row_to_give_it_to"),
        ],
    )
    def test_a_numbering_that_just_fits_is_still_produced(
        self,
        start: int,
        height: int,
        expected: list[int],
    ) -> None:
        """The boundary has to fall on the right side. A range check that refuses one number too
        many would break a legitimate numbering that ends on the largest id there is.

        The empty frame is the case worth reading twice. There is no last number when no row is
        numbered, so a start at the ceiling is fine, and the column still comes back typed.
        """
        frame = _rows().head(height)

        result = run_tool(RecordId(), Ports({"rows": frame}), {"output": "row_no", "start": start})

        produced = result["rows"]
        assert produced["row_no"].to_list() == expected
        assert produced.schema["row_no"] == pl.Int64, "the column is a 64-bit integer whatever the start"

    def test_the_column_is_a_64_bit_integer_however_large_the_start_written_was(self) -> None:
        """A start the column cannot hold must never turn the column into some other type.

        This is the failure that survives everything else. Handed a start past the ceiling, the
        engine widens the column to hold it, so a frame comes back carrying a decimal column where
        the contract says a 64-bit integer. The next join reads the declared type and finds another
        one. The refusal has to happen before that, which is what this checks from the other side.
        """
        frame = _rows()

        with pytest.raises(ConfigError):
            run_tool(RecordId(), Ports({"rows": frame}), {"output": "row_no", "start": _INT64_MAX + 1})

    def test_a_start_nothing_could_hold_is_refused_without_a_frame(self) -> None:
        """A start past the ceiling is wrong on its own, so it does not need rows to be refused.

        ``start + height - 1`` needs the frame, so that half belongs where the frame is. ``start``
        itself does not. Refusing it in the configuration phase is what lets a definition file be
        checked before anything is read.

        The second half of this says the opposite about a start that does fit. That one must get
        past the configuration phase, because whether it works depends on how many rows arrive, and
        refusing it early would reject a numbering that is perfectly fine on a one-row frame.
        """
        tool = RecordId()

        with pytest.raises(ConfigError) as caught:
            tool.validate_config({"output": "row_no", "start": _INT64_MAX + 1})

        assert "start" in str(caught.value), f"refusal should name the field to fix: {caught.value}"

        tool.validate_config({"output": "row_no", "start": _INT64_MAX})
