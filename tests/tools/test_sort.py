"""Tests for the Sort tool: which row comes first, and what must not change while it is decided.

Sort takes one input port, ``rows``, and produces one, ``rows``. The same rows come back, with the
same columns, in the same column order, holding the same types. Only the order of the rows moves.

Six decisions are pinned here, because each has a quiet failure mode.

**Each key carries its own direction and its own placement for missing values.** One list of keys
can say "zone ascending, missing zones first, then amount descending, missing amounts last". An
implementation that reads one direction and applies it to every key still looks right on a
single-key config, which is the config everybody tries first.

The placement half of that needs a frame built for it, and it has its own class below. A missing
value only shows where it was placed when other rows could have gone in its slot. On a second key
that means the missing value has to sit inside a tie group the first key left. Anywhere else it is
alone in its group, and its placement cannot change the answer no matter what the tool does with
it.

**Sort and record_id compose.** Sorting a frame and then numbering it has to give ids that count up
in arrival order within every tie group. That is the promise a pipeline leans on when it sorts and
then numbers, and neither tool's own file can check it alone. The composition test lives here,
because the thing that can break is the sort's stability; record_id has no order of its own to
lose.

**The sort is stable.** Rows that tie on every key keep the order they arrived in. This is the
reason the tool exists in the shape it does. A generated row number is built on top of a sort, so
an unstable sort makes that number depend on how the engine happened to lay the rows out rather
than on the data, and the same pipeline run twice produces two different answers.

Worth being honest about the limit of the test below: the query engine underneath happens to leave
ties alone today even when nothing asks it to, so the stability test pins the promise rather than
catching a missing flag. The engine makes no such promise, which is why the promise has to be
stated here and asked for in the tool. The catalog carries the same claim in a stronger form: one
worked example holds a deliberate tie and is compared with row order significant, so a tool that
reorders ties is caught by conformance as a drifted example.

**Only the row order moves.** The columns, their order and their types are untouched, and every row
is still there afterwards -- none dropped, none duplicated, none invented. A sort is the step a
reader trusts without looking, so the things it must not do are checked on every case.

**A frame with no rows sorts to a frame with no rows, schema intact.** A later step is compiled
against that schema whether or not any row ever arrived.

**Every refusal is read off the schema, not off the data.** Each refusal case runs twice: once on a
populated frame and once on a frame with no rows at all. Passing both is what shows the refusal came
from the column names and the configuration rather than from the values.

Vocabulary is deliberately generic -- zone, amount, count, code -- because the library carries no
caller's nouns.
"""

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Final

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.tool import (
    CATEGORIES,
    REGISTRY,
    ConfigError,
    ConformanceError,
    Ports,
    check_tool,
    expression_fields,
    run_tool,
)
from dantron_etl.tools.record_id import RecordId
from dantron_etl.tools.sort import Sort

# Values YAML turns into something other than the word written; no enum in this library may use one.
_YAML_UNSAFE: Final[frozenset[str]] = frozenset({"true", "false", "yes", "no", "on", "off", "null", "y", "n"})


def _rows() -> pl.DataFrame:
    """One frame carrying every ordering shape worth pinning, with each row labelled by its position.

    ``position`` is the row's place in the input, so an expected order can be written as a list of
    positions and read back off the output. Rows 0 and 3 are identical on ``zone`` and on ``amount``,
    which is the deliberate tie the stability claim needs. ``zone``, ``amount``, ``count`` and
    ``opened`` each carry a missing value, so where missing values land can be pinned on text, on a
    float, on an integer and on a date. ``code`` is distinct everywhere, for the case with no ties
    at all.
    """
    return pl.DataFrame(
        {
            "position": [0, 1, 2, 3, 4, 5],
            "zone": ["north", None, "east", "north", "south", None],
            "amount": [10.0, -5.0, None, 10.0, 2.5, 0.0],
            "count": [3, 1, 2, 3, None, 4],
            "opened": [
                date(2024, 3, 1),
                date(2024, 1, 1),
                None,
                date(2024, 3, 1),
                date(2024, 2, 1),
                date(2024, 1, 1),
            ],
            "code": ["a", "b", "c", "d", "e", "f"],
        }
    )


def _take(frame: pl.DataFrame, positions: Sequence[int]) -> pl.DataFrame:
    """The rows of ``frame`` at ``positions``, in that order, keeping the schema when there are none."""
    return frame[list(positions)] if positions else frame.head(0)


def _ordering_cases() -> list[Any]:
    """(config, the input positions the output must hold, in order) for each ordering worth pinning."""
    return [
        pytest.param(
            {"by": [{"column": "zone"}]},
            (2, 0, 3, 4, 1, 5),
            id="text_ascending_is_the_default_and_missing_values_go_last",
        ),
        pytest.param(
            {"by": [{"column": "zone", "direction": "desc"}]},
            (4, 0, 3, 2, 1, 5),
            id="text_descending_still_puts_missing_values_last",
        ),
        pytest.param(
            {"by": [{"column": "zone", "nulls": "first"}]},
            (1, 5, 2, 0, 3, 4),
            id="missing_values_first_when_asked_for",
        ),
        pytest.param(
            {"by": [{"column": "amount"}]},
            (1, 5, 4, 0, 3, 2),
            id="floats_ascending_including_a_negative_one",
        ),
        pytest.param(
            {"by": [{"column": "count", "direction": "desc", "nulls": "first"}]},
            (4, 5, 0, 3, 2, 1),
            id="integers_descending_with_the_missing_count_on_top",
        ),
        pytest.param(
            {"by": [{"column": "opened"}]},
            (1, 5, 4, 0, 3, 2),
            id="dates_ascending_in_calendar_order",
        ),
        pytest.param(
            {"by": [{"column": "code", "direction": "desc"}]},
            (5, 4, 3, 2, 1, 0),
            id="a_key_with_no_ties_and_no_missing_values",
        ),
        pytest.param(
            {"by": [{"column": "zone"}, {"column": "position", "direction": "desc"}]},
            (2, 3, 0, 4, 5, 1),
            id="a_second_key_breaks_the_ties_the_first_key_leaves",
        ),
        pytest.param(
            {
                "by": [
                    {"column": "zone", "direction": "asc", "nulls": "first"},
                    {"column": "amount", "direction": "desc", "nulls": "last"},
                ]
            },
            (5, 1, 2, 0, 3, 4),
            id="each_key_carries_its_own_direction_and_its_own_missing_value_placement",
        ),
    ]


class TestSortOrdering:
    """The rows move; nothing else does."""

    @pytest.mark.parametrize(("config", "expected"), _ordering_cases())
    def test_the_rows_come_back_in_the_order_asked_for(
        self,
        config: Mapping[str, Any],
        expected: tuple[int, ...],
    ) -> None:
        """Comparing against a slice of the input pins the order and everything that must not move.

        ``_take`` rebuilds the input in the expected order, so one comparison covers which rows, in
        which order, in which column order, with which types, holding which values. The two
        assertions after it say the rest out loud, because they are the things a reader assumes
        without checking: every row is still there, exactly once.
        """
        frame = _rows()

        result = run_tool(Sort(), Ports({"rows": frame}), config)

        assert result.names == ("rows",), "one input, one output, both named for what they carry"
        assert_frame_equal(result["rows"], _take(frame, expected))
        assert result["rows"].schema == frame.schema, "only the row order may move"
        assert sorted(result["rows"]["position"].to_list()) == list(range(frame.height)), (
            "no row may be dropped, duplicated or invented by an ordering"
        )

    @pytest.mark.parametrize(("config", "expected"), _ordering_cases())
    def test_a_frame_with_no_rows_sorts_to_a_frame_with_no_rows(
        self,
        config: Mapping[str, Any],
        expected: tuple[int, ...],
    ) -> None:
        """An empty frame keeps its schema, so the next step can still be compiled against it.

        Every ordering is run again with nothing to order, because a tool that reaches for the first
        row to decide something works perfectly until the day a filter upstream matches nothing.
        """
        empty = _rows().head(0)

        result = run_tool(Sort(), Ports({"rows": empty}), config)

        assert result["rows"].height == 0
        assert_frame_equal(result["rows"], empty)

    def test_rows_that_tie_on_every_key_keep_the_order_they_arrived_in(self) -> None:
        """The stability promise, on a frame built so that only stability can produce the answer.

        Every row shares one of three zones, and the position column counts up across the whole
        frame, so within any zone the positions must still count up afterwards. If ties were
        reordered, the positions inside a zone would come back shuffled.

        The same claim is made again at the other end of a two-key sort, where the tie is a tie on
        both keys rather than on one, because an implementation can be stable on the first key and
        lose the order while breaking ties on the second.
        """
        frame = pl.DataFrame(
            {
                "zone": [f"zone_{index % 3}" for index in range(300)],
                "band": [index % 2 for index in range(300)],
                "position": list(range(300)),
            }
        )

        one_key = run_tool(Sort(), Ports({"rows": frame}), {"by": [{"column": "zone"}]})
        two_keys = run_tool(
            Sort(),
            Ports({"rows": frame}),
            {"by": [{"column": "zone"}, {"column": "band", "direction": "desc"}]},
        )

        for result in (one_key, two_keys):
            assert result["rows"].height == frame.height
        for zone in ("zone_0", "zone_1", "zone_2"):
            within = one_key["rows"].filter(pl.col("zone") == zone)["position"].to_list()
            assert within == sorted(within), f"ties on {zone!r} were reordered: {within[:10]}"
        for zone in ("zone_0", "zone_1", "zone_2"):
            for band in (0, 1):
                within = two_keys["rows"].filter((pl.col("zone") == zone) & (pl.col("band") == band))
                positions = within["position"].to_list()
                assert positions == sorted(positions), (
                    f"ties on both keys ({zone!r}, band {band}) were reordered: {positions[:10]}"
                )

    def test_a_single_row_and_an_already_sorted_frame_come_back_unchanged(self) -> None:
        """The two shapes where the right answer is "do nothing", which is easy to get wrong.

        A sort that rebuilt its frame rather than reordering it would show up here as a changed
        dtype or a changed column order on the frame that needed no work at all.
        """
        frame = _rows()
        sorted_once = run_tool(Sort(), Ports({"rows": frame}), {"by": [{"column": "code"}]})

        sorted_twice = run_tool(Sort(), Ports({"rows": sorted_once["rows"]}), {"by": [{"column": "code"}]})
        single = run_tool(Sort(), Ports({"rows": frame.head(1)}), {"by": [{"column": "code"}]})

        assert_frame_equal(sorted_twice["rows"], sorted_once["rows"])
        assert_frame_equal(single["rows"], frame.head(1))

    def test_sort_leaves_its_input_untouched(self) -> None:
        """The boundary compares input schemas; the frame itself must be unchanged too.

        A sort that reordered its input in place would pass a boundary check that only compares
        schemas, and would silently reorder the other branch of any fan-out.
        """
        frame = _rows()
        before = frame.clone()

        run_tool(Sort(), Ports({"rows": frame}), {"by": [{"column": "amount", "direction": "desc"}]})

        assert_frame_equal(frame, before)


def _refusal_cases() -> list[Any]:
    """(config, the text its refusal must quote) for each way a list of sort keys can be wrong."""
    return [
        pytest.param({"by": []}, "config field 'by' must not be empty", id="no_sort_keys_at_all"),
        pytest.param({"by": None}, "required config field 'by'", id="a_by_written_with_no_value"),
        pytest.param({"by": [{"column": "quantity"}]}, "quantity", id="a_column_that_is_not_there"),
        pytest.param({"by": [{"column": "Zone"}]}, "Zone", id="a_column_whose_case_is_wrong"),
        pytest.param(
            {"by": [{"column": "zone"}, {"column": "amount"}, {"column": "zone"}]},
            "by[2]",
            id="the_same_column_named_twice",
        ),
        pytest.param({"by": [{"column": ""}]}, "by[0].column", id="a_column_name_that_is_empty"),
        pytest.param({"by": [{"column": "   "}]}, "column", id="a_column_name_that_is_only_whitespace"),
        pytest.param(
            {"by": [{"column": "zone", "direction": "ascending"}]},
            "ascending",
            id="a_direction_spelled_out_in_full",
        ),
        pytest.param(
            {"by": [{"column": "zone", "direction": "up"}]},
            "up",
            id="a_direction_nobody_published",
        ),
        pytest.param({"by": [{"column": "zone", "nulls": "top"}]}, "top", id="a_nulls_placement_nobody_published"),
        pytest.param(
            {"by": [{"column": "zone", "direction": None}]},
            "direction",
            id="a_direction_written_with_no_value",
        ),
        pytest.param({"by": [{"column": "zone", "nulls": None}]}, "nulls", id="a_nulls_written_with_no_value"),
        pytest.param({"by": [{"column": "zone", "order": "desc"}]}, "by[0].order", id="a_key_member_nobody_declared"),
        pytest.param({"by": [{"direction": "desc"}]}, "by[0].column", id="a_key_with_no_column_at_all"),
        pytest.param({"by": [{"column": 7}]}, "by[0].column", id="a_column_name_that_is_not_text"),
        pytest.param({"by": ["zone"]}, "by[0]", id="a_bare_column_name_where_a_key_belongs"),
    ]


class TestSortRefusals:
    """What Sort refuses, and when: in its own error family, off the schema, before any row moves."""

    @pytest.mark.parametrize(("config", "named"), _refusal_cases())
    @pytest.mark.parametrize("height", [6, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_list_of_keys_the_schema_cannot_satisfy_is_refused_by_name(
        self,
        config: Mapping[str, Any],
        named: str,
        height: int,
    ) -> None:
        """Every one of these has to arrive as a :class:`ConfigError`, never as an engine exception.

        A caller of this library catches one family. A column that is not there, an option nobody
        published, a key written as a bare name -- left to the query engine, each of those surfaces
        as some other library's exception type naming neither the field to open nor the file to fix.

        The same refusal has to hold on a frame with no rows, which is what shows it was read off
        the column names and the configuration rather than off the values.

        A direction or a nulls placement written with no value is the case worth reading twice. A
        definition file spells that ``direction:``, which arrives here as nothing at all, and taking
        it as "not configured" would run the sort under a policy its author never chose and was
        never told about.
        """
        frame = _rows().head(height)

        with pytest.raises(ConfigError) as caught:
            run_tool(Sort(), Ports({"rows": frame}), config)

        message = str(caught.value)
        assert named in message, f"refusal should name {named!r}: {message}"

    def test_a_repeated_column_is_refused_naming_both_places_it_was_written(self) -> None:
        """The refusal has to point at both keys, not just say the column is a duplicate.

        Picture a file with eight sort keys. Being told that 'zone' appears twice still leaves the
        reader scanning all eight lines. Being told it appears at ``by[2]`` and at ``by[0]`` sends
        them straight to the two lines that matter.

        The other half of this is the reason. A repeated key is refused because the second one can
        never decide anything, not because repeats are untidy. Pinning a phrase from that sentence
        is what stops the message shrinking to "duplicate column" later on.
        """
        frame = _rows()
        config = {"by": [{"column": "zone"}, {"column": "amount"}, {"column": "zone"}]}

        with pytest.raises(ConfigError) as caught:
            run_tool(Sort(), Ports({"rows": frame}), config)

        message = str(caught.value)
        assert "zone" in message, f"refusal should name the repeated column: {message}"
        assert "by[0]" in message, f"refusal should name the first place it was written: {message}"
        assert "by[2]" in message, f"refusal should name the second place it was written: {message}"
        assert "amount" not in message, (
            f"the key that is not a duplicate has nothing to do with this refusal: {message}"
        )
        assert "can never decide anything" in message, (
            f"the refusal should say why a repeated key is useless, not just that it is repeated: {message}"
        )

    def test_the_template_declares_a_typed_list_of_keys_with_yaml_safe_options(self) -> None:
        """``items`` and ``fields`` are what make the members of a key real rather than decorative.

        Without them, ``by`` is one untyped array: a misspelled member name is accepted, a numeric
        column name reaches the engine, and a direction nobody published is never compared against
        anything.

        The option words have to survive a definition file. ``asc``, ``desc``, ``first`` and ``last``
        all do. The obvious alternatives do not: a file spelling ``no`` hands over a boolean, and a
        file spelling ``null`` hands over nothing, which would be read as a field nobody configured.

        The default direction and the default placement are deliberately not declared on the nested
        fields. The contract fills defaults at the top level of a configuration only, so a default
        declared inside a key would read as if it applied and never be applied -- which the catalog
        check refuses outright. The tool supplies them in its own code, and the behaviour tests above
        are what prove it does.
        """
        template = Sort().config_template()

        assert set(template) == {"by"}, "one field; a sort has nothing else to say"
        assert template["by"].type == "array"
        assert template["by"].required is True
        assert template["by"].non_empty is True, "an empty list is a value, so 'required' alone accepts it"

        key = template["by"].items
        assert key is not None, "an array with no declared element type checks nothing"
        assert key.type == "object"
        assert key.fields is not None, "an object with no declared members checks nothing"
        assert set(key.fields) == {"column", "direction", "nulls"}
        assert key.fields["column"].type == "string"
        assert key.fields["column"].required is True
        assert key.fields["direction"].required is False, "a direction left out means ascending"
        assert key.fields["nulls"].required is False, "a placement left out means last"
        assert tuple(key.fields["direction"].enum or ()) == ("asc", "desc")
        assert tuple(key.fields["nulls"].enum or ()) == ("first", "last")
        for member in ("direction", "nulls"):
            offered = {value.lower() for value in key.fields[member].enum or ()}
            assert not (offered & _YAML_UNSAFE), f"{member} offers a word a definition file rewrites"
            assert key.fields[member].default is None, (
                "a default declared below the top level is never applied, and the catalog check "
                "refuses one; the tool has to supply it in its own code"
            )
        assert expression_fields(template) == (), "a sort key is a column name, not a computed value"


class TestSortInTheCatalog:
    """Described, documented with examples that actually run, and resolvable by name."""

    def test_sort_conforms_and_its_examples_show_a_direction_and_more_than_one_key(self) -> None:
        """Conformance runs every worked example, so a documented ordering is a checked ordering."""
        tool = Sort()

        check_tool(tool)

        assert tool.name == "sort"
        assert tool.category == "preparation", "changing the shape of a frame without touching a value"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",)

        examples = tool.help().examples
        assert examples, "a tool nobody can copy from is a tool nobody uses"
        assert any(
            any(key.get("direction") == "desc" for key in example.configuration["by"]) for example in examples
        ), "descending is the setting most likely to be copied, so it must be run"
        assert any(len(example.configuration["by"]) > 1 for example in examples), (
            "more than one key, with the second breaking the first's ties, is the shape the prose "
            "claims and an example has to demonstrate"
        )

    def test_sort_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in tests/core/test_registry.py."""
        assert "sort" in REGISTRY
        assert isinstance(REGISTRY.get("sort"), Sort)


class _ReversingSort(Sort):
    """The tool, ordered correctly and then handed back backwards.

    Standing in for an implementation that loses the order of rows that tie, in a form that fails
    the same way on every run rather than however the engine happens to lay out a small frame.
    """

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        produced = super().execute(inputs, config)
        return Ports({"rows": produced["rows"].reverse()})


class TestTheStabilityClaimIsCheckedInTheCatalog:
    """A reader deciding whether to trust the stability claim reads the worked example, not this file.

    Conformance compares an example without regard to row order unless the example asks, so for a
    tool whose whole output is an order, asking is not optional -- and asking has to be worth
    something. One example must also hold a genuine tie, or the claim it carries is untested even
    with the order comparison switched on.
    """

    def test_a_worked_example_holds_a_tie_declares_its_order_significant_and_catches_a_reordering(
        self,
    ) -> None:
        """The tie is the point. Without one, an ordered comparison proves only that sorting sorts."""
        examples = Sort().help().examples

        assert all(example.check_row_order for example in examples), (
            "row order is this tool's entire output; an example compared without it proves nothing"
        )

        tied = []
        for example in examples:
            columns = [key["column"] for key in example.configuration["by"]]
            frame = example.inputs["rows"]
            if frame.height != frame.select(columns).unique().height:
                tied.append(example.scenario)
        assert tied, (
            "no worked example holds two rows that tie on every sort key, so the catalog cannot "
            "show what happens to a tie"
        )

        check_tool(Sort())

        with pytest.raises(ConformanceError) as caught:
            check_tool(_ReversingSort())

        message = str(caught.value)
        assert "sort" in message, f"refusal should name the tool: {message}"
        assert "drifted" in message, f"refusal should say the example no longer matches the tool: {message}"


def _placement_rows() -> pl.DataFrame:
    """A frame where the placement chosen for the second key actually changes the answer.

    Rows 0, 1 and 2 all sit in the same zone. That is the tie group. Row 1 is the one with no
    amount, so it lives inside that group alongside two rows that do have one. Wherever the second
    key sends its missing values, row 1 has somewhere to go and two rows to move past.

    Row 3 has no zone, so the first key's own placement is visible too. Row 4 gives the first key a
    second real value to order against.
    """
    return pl.DataFrame(
        {
            "position": [0, 1, 2, 3, 4],
            "zone": ["north", "north", "north", None, "south"],
            "amount": [5.0, None, 1.0, 9.0, 2.0],
        }
    )


def _three_key_placement_rows() -> pl.DataFrame:
    """The same idea one level deeper, so a middle key's placement has to be read as its own.

    Rows 0 to 3 share a zone. Row 1 has no band, so it sits inside that group against rows that do.
    Rows 0 and 2 then share a band as well, and row 2 has no amount, so it sits inside that smaller
    group against a row that does.

    Row 4 has no zone. Every one of the three keys therefore holds a missing value whose placement
    can be seen in the output.
    """
    return pl.DataFrame(
        {
            "position": [0, 1, 2, 3, 4, 5],
            "zone": ["north", "north", "north", "north", None, "south"],
            "band": [1, None, 1, 2, 1, 1],
            "amount": [5.0, 3.0, None, 4.0, 1.0, 2.0],
        }
    )


class TestEachSortKeyPlacesItsOwnMissingValues:
    """The placement written on a key applies to that key and to no other key.

    A tool that read one placement and applied it to every key would pass every single-key case. It
    would also pass every multi-key case where the later keys have no missing value inside a tie
    group. Those are the easy cases to write, so they are the cases a test file fills up with.

    Each case below carries opposite placements on neighbouring keys. That is what makes a spread
    placement visible in either direction. A tool that pushed the first key's placement across the
    list gets one answer, a tool that pushed the last key's across gets another, and both differ
    from the answer written down here.
    """

    def test_a_missing_value_on_the_second_key_goes_where_that_key_says(self) -> None:
        """First key puts missing values first, second key puts them last. They do not agree.

        The zone key sends row 3, which has no zone, to the top. Inside the northern tie group the
        amount key sends row 1, which has no amount, to the bottom.

        Push the first key's 'first' onto both and row 1 would lead the northern group instead of
        trailing it. Push the second key's 'last' onto both and row 3 would fall to the end of the
        frame.
        """
        frame = _placement_rows()
        config = {
            "by": [
                {"column": "zone", "nulls": "first"},
                {"column": "amount", "nulls": "last"},
            ]
        }

        result = run_tool(Sort(), Ports({"rows": frame}), config)

        assert_frame_equal(result["rows"], _take(frame, (3, 2, 0, 1, 4)))

    def test_the_same_pair_of_placements_the_other_way_round(self) -> None:
        """First key puts missing values last, second key puts them first. The mirror of the above.

        One of these two cases dies when the first key's placement is spread across the list. The
        other dies when the last key's is. Having only one of them would leave the other spread
        alive, because a single case cannot say which end of the list a shared placement came from.
        """
        frame = _placement_rows()
        config = {
            "by": [
                {"column": "zone", "nulls": "last"},
                {"column": "amount", "nulls": "first"},
            ]
        }

        result = run_tool(Sort(), Ports({"rows": frame}), config)

        assert_frame_equal(result["rows"], _take(frame, (1, 2, 0, 4, 3)))

    def test_three_keys_with_three_placements_are_read_one_by_one(self) -> None:
        """Last, then first, then last. No single placement can stand in for the other two.

        The zone key keeps row 4, which has no zone, at the bottom. The band key lifts row 1, which
        has no band, to the top of the northern group. The amount key drops row 2, which has no
        amount, below row 0 inside the smaller group the first two keys left.

        This case also kills a middle key spread across the list, which the two-key cases cannot
        reach. Push the band key's 'first' onto the others and row 4 jumps to the front of the frame.
        """
        frame = _three_key_placement_rows()
        config = {
            "by": [
                {"column": "zone", "nulls": "last"},
                {"column": "band", "nulls": "first"},
                {"column": "amount", "nulls": "last"},
            ]
        }

        result = run_tool(Sort(), Ports({"rows": frame}), config)

        assert_frame_equal(result["rows"], _take(frame, (1, 0, 2, 3, 5, 4)))

    def test_a_placement_written_on_one_key_does_not_reach_a_key_that_wrote_none(self) -> None:
        """A key that says nothing takes the default, whatever its neighbour asked for.

        The zone key asks for missing values first. The amount key asks for nothing, so its missing
        values go last. A tool that let a written placement leak onto a silent key would put row 1
        at the head of the northern group.
        """
        frame = _placement_rows()
        config = {"by": [{"column": "zone", "nulls": "first"}, {"column": "amount"}]}

        result = run_tool(Sort(), Ports({"rows": frame}), config)

        assert_frame_equal(result["rows"], _take(frame, (3, 2, 0, 1, 4)))


def _flat_key_rows(height: int, value: str | None) -> pl.DataFrame:
    """A tall frame whose sort key holds one value, or no value, on every row.

    ``position`` counts up across the whole frame. It is the only thing telling one row from
    another, and it is never a sort key, so any reordering shows up in it.
    """
    return pl.DataFrame(
        {
            "position": list(range(height)),
            "period": pl.Series([value] * height, dtype=pl.String),
            "amount": [float(index) for index in range(height)],
        }
    )


class TestSortOnAKeyThatSeparatesNothing:
    """Sorting on a column holding one value, or none, has to hand the frame back as it was.

    This is a real shape, not a contrived one. A column that is absent across a whole batch arrives
    as a key of nothing but missing values. Sorting on it is the case where every row ties with
    every other row. There is exactly one correct answer: the input, unchanged.

    These pin the contract at that shape. They cannot catch a missing stability flag, and they are
    not here to. The engine leaves ties alone here either way, which the prose at the top of this
    file already says. What they are good for is fixing the answer, so a later change that starts
    shuffling ties on a flat key has something to fail against.
    """

    @pytest.mark.parametrize("nulls", ["last", "first"])
    def test_a_key_that_is_missing_on_every_row_leaves_the_frame_alone(self, nulls: str) -> None:
        """Every row ties, so the output is the input. Both placements give the same answer.

        Where missing values go stops meaning anything when they are the only values there are. A
        tool that took the all-missing case down a path of its own would show it here, and the two
        placements would disagree.
        """
        frame = _flat_key_rows(1000, None)

        result = run_tool(Sort(), Ports({"rows": frame}), {"by": [{"column": "period", "nulls": nulls}]})

        assert_frame_equal(result["rows"], frame)
        assert result["rows"]["position"].to_list() == list(range(1000)), "the rows arrived in this order"

    @pytest.mark.parametrize("direction", ["asc", "desc"])
    def test_a_key_holding_one_repeated_value_leaves_the_frame_alone(self, direction: str) -> None:
        """The same frame with a value in the key instead of a gap. The answer does not change.

        A thousand identical values sorted ascending and sorted descending give the same thousand
        rows in the same order. Nothing in the key separates them, so the direction has nothing to
        act on.
        """
        frame = _flat_key_rows(1000, "period_one")

        result = run_tool(
            Sort(),
            Ports({"rows": frame}),
            {"by": [{"column": "period", "direction": direction}]},
        )

        assert_frame_equal(result["rows"], frame)
        assert result["rows"]["position"].to_list() == list(range(1000)), "the rows arrived in this order"


class TestSortFeedingRecordId:
    """Sort, then number. The ids inside a tie group have to follow arrival order.

    This lives in the sort file rather than the record_id file because sort is the tool that can
    break it. RecordId numbers rows where it finds them, so it has no order of its own to lose. If
    the ids inside a tie group come back scrambled, the sort reordered the tie. Putting the test in
    record_id's file would send whoever reads the failure to the wrong module.

    One thing this deliberately does not do is run the pair twice and compare the runs. The query
    engine is deterministic for a given input, so two runs agree whether the sort is stable or not.
    Agreement between runs would prove nothing. The assertion is against arrival order, which is a
    fact about the input rather than about a previous run.
    """

    def test_ids_count_up_in_arrival_order_inside_every_tie_group(self) -> None:
        """A tie-heavy frame, sorted then numbered, with each group checked against how it arrived.

        Four zones, a hundred rows each, and an ``arrival`` column counting up across all four
        hundred. Sorting on zone ties every row with ninety-nine others, so the sort has four
        hundred rows and almost nothing to tell them apart by.

        After numbering, reading one zone's rows off the output in id order has to give its arrival
        numbers counting up. That is the whole promise. The id a row ends up with is decided by the
        order the rows came in, not by how the engine laid them out.
        """
        height = 400
        frame = pl.DataFrame(
            {
                "zone": [f"zone_{index % 4}" for index in range(height)],
                "arrival": list(range(height)),
            }
        )

        ordered = run_tool(Sort(), Ports({"rows": frame}), {"by": [{"column": "zone"}]})
        numbered = run_tool(RecordId(), Ports({"rows": ordered["rows"]}), {"output": "row_no"})

        produced = numbered["rows"]
        assert produced.height == height, "sorting and numbering adds no rows and removes none"
        assert produced["row_no"].to_list() == list(range(1, height + 1)), (
            "the numbering follows the order the sort produced, one row at a time"
        )

        for zone in ("zone_0", "zone_1", "zone_2", "zone_3"):
            group = produced.filter(pl.col("zone") == zone)
            ids = group["row_no"].to_list()
            arrivals = group["arrival"].to_list()
            assert arrivals == sorted(arrivals), (
                f"ids inside {zone!r} do not follow arrival order: {list(zip(ids, arrivals, strict=True))[:10]}"
            )
            assert ids == list(range(ids[0], ids[0] + len(ids))), (
                f"a zone's rows land together, so its ids should run without a gap: {ids[:10]}"
            )

    def test_the_ids_a_row_gets_follow_the_sort_that_came_before_them(self) -> None:
        """Change the sort and the ids move with it. That is why the pair is worth composing.

        The same frame numbered after two different sorts gives a row two different ids. If it gave
        the same id either way, the numbering would be following something other than the order the
        sort chose, and putting a sort in front of it would buy nothing.
        """
        frame = pl.DataFrame({"code": ["c", "a", "b"], "amount": [3.0, 1.0, 2.0]})

        by_code = run_tool(Sort(), Ports({"rows": frame}), {"by": [{"column": "code"}]})
        by_amount = run_tool(Sort(), Ports({"rows": frame}), {"by": [{"column": "amount", "direction": "desc"}]})

        ascending = run_tool(RecordId(), Ports({"rows": by_code["rows"]}), {"output": "row_no"})["rows"]
        descending = run_tool(RecordId(), Ports({"rows": by_amount["rows"]}), {"output": "row_no"})["rows"]

        assert dict(zip(ascending["code"], ascending["row_no"], strict=True)) == {"a": 1, "b": 2, "c": 3}
        assert dict(zip(descending["code"], descending["row_no"], strict=True)) == {"c": 1, "b": 2, "a": 3}
