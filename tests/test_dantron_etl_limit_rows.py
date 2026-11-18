"""Tests for the LimitRows tool: keep the first N rows, and hand back the rest rather than dropping them.

LimitRows takes one input port, ``rows``, and produces two: ``rows`` carries the first ``count``
rows, and ``dropped`` carries everything after them.

Two output ports is the whole design. A tool that quietly discards rows is how a report comes to
show the top five hundred of something while reading as though it showed everything. So the rest of
the frame leaves by a named port, with the input's schema exactly, and a flow that ignores it is
ignoring it on purpose. Five decisions are pinned here.

**The first ``count`` rows in the order they arrived.** This tool never sorts. The frame below is
deliberately out of order, so an implementation that sorts first -- which is what "top N" means
almost everywhere else -- hands back different rows and dies here.

**``rows`` and ``dropped`` together are the input.** In order, nothing lost, nothing duplicated,
nothing invented. Stacking the two ports back up has to give the frame that went in. That is the
property that makes the second port worth having: without it, the tool is a discard with extra
steps.

**A frame shorter than ``count`` comes back whole, and ``dropped`` is empty but typed.** An empty
port still carries its columns, because a later step is compiled against that schema whether or not
a row ever reaches it. The contract already refuses an output port with no columns at all, so a
tool that built its empty side out of nothing would be caught -- but the types have to survive too,
and only a comparison catches that.

**``count: 0`` is legal.** Keep none is a real thing to configure, and refusing it would be a
special case with no reason behind it. The count is often a parameter, and a parameter that reaches
zero should behave like the limit it is rather than stopping the run. Nothing is lost when it does,
because every row leaves by ``dropped``.

**Columns, column order and types are unchanged on both ports.** This tool decides which rows
leave by which door and nothing else.

**Every refusal is read off the configuration, not off the data.** Each refusal case runs twice:
once on a populated frame and once on a frame with no rows at all. A count written with no value is
the one worth reading twice -- a definition file spells that ``count:``, it arrives as nothing at
all, and read as "not configured" it would run under a limit its author never chose and was never
told about.

Vocabulary is deliberately generic -- code, amount, position -- because the library carries no
caller's nouns.
"""

from collections.abc import Mapping, Sequence
from typing import Any

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
from dantron_etl.tools.limit_rows import LimitRows


def _rows() -> pl.DataFrame:
    """Five rows, deliberately not in order by anything.

    ``position`` is the row's place in the input, so an expected split can be written as two lists
    of positions and read back off the two ports. ``amount`` counts neither up nor down, and
    ``code`` is not alphabetical, so a tool that sorts by either before taking its slice keeps a
    different set of rows. ``opened`` and ``count`` are there so more than one type has to survive
    the trip.
    """
    return pl.DataFrame(
        {
            "position": [0, 1, 2, 3, 4],
            "code": ["d", "a", "e", "b", "c"],
            "amount": [50.0, 10.0, 90.0, None, 70.0],
            "count": [3, 1, None, 4, 2],
        },
        schema={"position": pl.Int64, "code": pl.String, "amount": pl.Float64, "count": pl.Int64},
    )


def _take(frame: pl.DataFrame, positions: Sequence[int]) -> pl.DataFrame:
    """The rows of ``frame`` at ``positions``, in that order, keeping the schema when there are none."""
    return frame[list(positions)] if positions else frame.head(0)


def _split_cases() -> list[Any]:
    """(count, the positions kept, the positions dropped) for each split worth pinning."""
    return [
        pytest.param(0, (), (0, 1, 2, 3, 4), id="keeping_none_sends_every_row_to_dropped"),
        pytest.param(1, (0,), (1, 2, 3, 4), id="keeping_one_takes_the_row_that_arrived_first"),
        pytest.param(3, (0, 1, 2), (3, 4), id="keeping_three_takes_the_first_three_as_they_arrived"),
        pytest.param(4, (0, 1, 2, 3), (4,), id="keeping_all_but_one"),
        pytest.param(5, (0, 1, 2, 3, 4), (), id="a_count_that_is_exactly_the_height_drops_nothing"),
        pytest.param(6, (0, 1, 2, 3, 4), (), id="a_count_one_past_the_height_drops_nothing"),
        pytest.param(1000, (0, 1, 2, 3, 4), (), id="a_count_far_past_the_height_drops_nothing"),
    ]


class TestLimitRowsSplitsTheFrameInTwo:
    """The first ``count`` rows leave by ``rows``; the rest leave by ``dropped``."""

    @pytest.mark.parametrize(("count", "kept", "discarded"), _split_cases())
    def test_both_ports_hold_the_rows_written_down_in_the_order_they_arrived(
        self,
        count: int,
        kept: tuple[int, ...],
        discarded: tuple[int, ...],
    ) -> None:
        """Comparing against slices of the input pins the split and everything that must not move.

        ``_take`` rebuilds the input at the expected positions, so one comparison per port covers
        which rows, in which order, in which column order, holding which types and which values.
        ``assert_frame_equal`` compares row order, which is the point: this tool's whole contract is
        about which rows come first.
        """
        frame = _rows()

        result = run_tool(LimitRows(), Ports({"rows": frame}), {"count": count})

        assert result.names == ("rows", "dropped"), "the rest of the frame leaves by a named port"
        assert_frame_equal(result["rows"], _take(frame, kept))
        assert_frame_equal(result["dropped"], _take(frame, discarded))

    @pytest.mark.parametrize(("count", "kept", "discarded"), _split_cases())
    def test_the_two_ports_stacked_back_up_are_the_input(
        self,
        count: int,
        kept: tuple[int, ...],
        discarded: tuple[int, ...],
    ) -> None:
        """Nothing lost, nothing duplicated, nothing invented, and the order intact.

        This is the property that makes the second port worth having. Stated separately from the
        comparison above because it is the claim a reader of a pipeline actually leans on: whatever
        the count, the rows are all still somewhere.
        """
        frame = _rows()

        result = run_tool(LimitRows(), Ports({"rows": frame}), {"count": count})

        assert_frame_equal(pl.concat([result["rows"], result["dropped"]]), frame)
        assert result["rows"].height + result["dropped"].height == frame.height
        assert len(kept) + len(discarded) == frame.height, "the case table itself has to account for every row"

    @pytest.mark.parametrize(("count", "kept", "discarded"), _split_cases())
    def test_both_ports_keep_the_columns_their_order_and_their_types(
        self,
        count: int,
        kept: tuple[int, ...],
        discarded: tuple[int, ...],
    ) -> None:
        """A port with no rows still has to carry the schema, or the next step cannot be compiled.

        The contract refuses an output port with no columns at all, so a port built out of nothing
        is already caught. What is not caught is a port rebuilt with the right column names and the
        wrong types, which is why this compares schemas rather than counting columns.
        """
        frame = _rows()

        result = run_tool(LimitRows(), Ports({"rows": frame}), {"count": count})

        for port in ("rows", "dropped"):
            assert result[port].schema == frame.schema, f"port {port!r} changed the shape of the frame"

    def test_limit_rows_never_sorts(self) -> None:
        """The first rows as they arrived, not the smallest or the first alphabetically.

        "Top N" means a sort followed by a limit almost everywhere else, and that is exactly the
        implementation somebody reaches for here. It is the wrong one. A sort is its own step, it
        runs in front of this one when a flow wants it, and its options belong to it.

        Three assertions rather than one, so a tool that sorted on any of the three columns present
        fails with a message that says which.
        """
        frame = _rows()

        result = run_tool(LimitRows(), Ports({"rows": frame}), {"count": 3})

        assert result["rows"]["position"].to_list() == [0, 1, 2], "the first three rows, as they arrived"
        assert result["rows"]["code"].to_list() == ["d", "a", "e"], "not the first three codes in order"
        assert result["rows"]["amount"].to_list() == [50.0, 10.0, 90.0], "not the three smallest amounts"

    @pytest.mark.parametrize(("count", "kept", "discarded"), _split_cases())
    def test_a_frame_with_no_rows_gives_two_frames_with_no_rows(
        self,
        count: int,
        kept: tuple[int, ...],
        discarded: tuple[int, ...],
    ) -> None:
        """Nothing to keep and nothing to drop, and both ports still carry the schema.

        Every count is run again with nothing to limit, because a tool that reaches for a row to
        decide something works perfectly until the day a filter upstream matches nothing.
        """
        empty = _rows().head(0)

        result = run_tool(LimitRows(), Ports({"rows": empty}), {"count": count})

        for port in ("rows", "dropped"):
            assert_frame_equal(result[port], empty)

    def test_limit_rows_leaves_its_input_untouched(self) -> None:
        """The boundary compares input schemas; the frame itself must be unchanged too.

        A tool that truncated its input in place would pass a check that only compares schemas, and
        would silently shorten the other branch of any fan-out.
        """
        frame = _rows()
        before = frame.clone()

        run_tool(LimitRows(), Ports({"rows": frame}), {"count": 2})

        assert_frame_equal(frame, before)


class TestKeepingNoneIsLegal:
    """``count: 0`` runs. It is a limit like any other, and nothing is lost when it is reached.

    Refusing zero would be a special case, and a special case needs a reason. There is none here.
    Keep none is a thing an author can mean, a count is often a parameter, and a parameter that
    arrives as zero should do what it says rather than stop the run. The row that would have
    justified refusing -- one that vanishes without trace -- does not exist, because every row
    leaves by ``dropped``.
    """

    def test_a_count_of_zero_empties_the_kept_port_and_fills_the_dropped_one(self) -> None:
        """Both halves of the claim: ``rows`` is empty with its schema, ``dropped`` is the input."""
        frame = _rows()

        result = run_tool(LimitRows(), Ports({"rows": frame}), {"count": 0})

        assert result["rows"].height == 0
        assert result["rows"].schema == frame.schema, "an empty port still carries its columns"
        assert_frame_equal(result["dropped"], frame)


def _refusal_cases() -> list[Any]:
    """(config, the text its refusal must quote) for each way a count can be wrong."""
    return [
        pytest.param({"count": -1}, "-1", id="a_negative_count"),
        pytest.param({"count": -1000}, "-1000", id="a_very_negative_count"),
        pytest.param({"count": None}, "count", id="a_count_written_with_no_value"),
        pytest.param({}, "count", id="no_count_at_all"),
        pytest.param({"count": 2.0}, "count", id="a_count_written_as_a_whole_looking_decimal"),
        pytest.param({"count": 2.5}, "count", id="a_count_written_as_a_fraction"),
        pytest.param({"count": "3"}, "count", id="a_count_written_as_text"),
        pytest.param({"count": True}, "count", id="a_count_written_as_a_truth_value"),
        pytest.param({"count": [3]}, "count", id="a_count_written_as_a_list"),
        pytest.param({"count": 3, "rows": 3}, "unknown config field", id="a_field_nobody_declared"),
        pytest.param({"limit": 3}, "limit", id="the_field_given_a_name_nobody_declared"),
    ]


class TestLimitRowsRefusals:
    """What LimitRows refuses, and when: in its own error family, before any row moves."""

    @pytest.mark.parametrize(("config", "named"), _refusal_cases())
    @pytest.mark.parametrize("height", [5, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_count_that_cannot_be_a_count_is_refused_by_name(
        self,
        config: Mapping[str, Any],
        named: str,
        height: int,
    ) -> None:
        """Every one of these has to arrive as a :class:`ConfigError`, never as an engine exception.

        A caller of this library catches one family. A negative count, a count written as text, a
        count written as a decimal -- left to the query engine, each of those surfaces as some other
        library's exception naming neither the field to open nor the file to fix.

        The count written with no value is the case worth reading twice. A definition file spells it
        ``count:``, which arrives here as nothing at all, and defaults are filled only for fields
        that are absent -- so a field written with no value is not a field nobody wrote, and taking
        it as one would run the step under a limit its author never chose.

        A truth value is on the list because a whole number in Python accepts one. ``count: true``
        would otherwise be read as keeping one row.
        """
        frame = _rows().head(height)

        with pytest.raises(ConfigError) as caught:
            run_tool(LimitRows(), Ports({"rows": frame}), config)

        message = str(caught.value)
        assert named in message, f"refusal should name {named!r}: {message}"

    def test_a_negative_count_is_refused_saying_what_to_write_instead(self) -> None:
        """The refusal quotes the number and points at the count that does mean "keep none".

        Somebody writing a negative count meant one of two things: an off-by-one in whatever
        produced it, or "keep nothing". Naming zero in the message settles the second case without a
        trip to the documentation, and quoting the offending number settles the first.
        """
        frame = _rows()

        with pytest.raises(ConfigError) as caught:
            run_tool(LimitRows(), Ports({"rows": frame}), {"count": -2})

        message = str(caught.value)
        assert "count" in message, f"refusal should name the field: {message}"
        assert "-2" in message, f"refusal should quote the number it refused: {message}"
        assert "0" in message, f"refusal should point at the count that keeps none: {message}"

    def test_the_template_declares_one_required_whole_number(self) -> None:
        """One field, and it is a whole number, so a decimal never reaches the slicing.

        A count declared as a plain number would accept 2.5 and leave the tool to round it, which is
        a policy nobody wrote down. Declared as an integer, the boundary refuses it and the author is
        told which field to fix.

        No default. A limit nobody configured is not a limit, and inventing one -- a thousand, say --
        would truncate a frame silently, which is the failure this tool's second port exists to make
        impossible.
        """
        template = LimitRows().config_template()

        assert set(template) == {"count"}, "one field; a limit has nothing else to say"
        assert template["count"].type == "integer"
        assert template["count"].required is True
        assert template["count"].default is None, "a limit nobody wrote is not a limit"
        assert expression_fields(template) == (), "a count is a number, not a computed value"


class TestLimitRowsInTheCatalog:
    """Described, documented with examples that actually run, and resolvable by name."""

    def test_limit_rows_conforms_and_an_example_shows_what_was_dropped(self) -> None:
        """Conformance runs every worked example, so a documented split is a checked split.

        Two extra demands. An example has to declare its row order significant, because which rows
        come first is this tool's entire output and a comparison that ignores order proves only that
        the right number of rows came back. And an example has to show a non-empty ``dropped``,
        because a catalog page whose every example drops nothing teaches a reader that the port is
        decoration.
        """
        tool = LimitRows()

        check_tool(tool)

        assert tool.name == "limit_rows"
        assert tool.category == "preparation", "which rows travel on, with no value changed"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows", "dropped")

        examples = tool.help().examples
        assert examples, "a tool nobody can copy from is a tool nobody uses"
        assert any(example.check_row_order for example in examples), (
            "which row comes first is this tool's output; an example compared without it proves little"
        )
        assert any(example.outputs["dropped"].height > 0 for example in examples), (
            "no worked example shows a row leaving by 'dropped', which is the reason the port exists"
        )

    def test_limit_rows_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in test_dantron_etl_registry.py."""
        assert "limit_rows" in REGISTRY
        assert isinstance(REGISTRY.get("limit_rows"), LimitRows)
