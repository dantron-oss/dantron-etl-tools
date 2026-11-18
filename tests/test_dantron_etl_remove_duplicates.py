"""Tests for the RemoveDuplicates tool: one input, two outputs, and nothing dropped between them.

RemoveDuplicates takes one input port, ``rows``, and produces two: ``unique``, carrying one member
of every group of identical rows, and ``duplicate``, carrying the members it did not pick. The
second port is the whole design decision. The tool this one is derived from had a ``keep`` option
whose third value meant "drop every row that has a duplicate", which is not a mode at all -- it is a
second output, asked for by an author who had nowhere to route the rows they were deleting.

Five decisions are pinned here, because each has a silent failure mode:

**Row conservation.** ``unique.height + duplicate.height == rows.height`` for every case, nulls
included. That is the invariant that makes "nothing is deleted, only routed" a fact rather than a
claim, and a de-duplication step is precisely where a caller most needs to see what went missing --
it is the step that exists to remove rows.

**``keep: first`` and ``keep: last`` are complements, not variations.** Each puts a *different*
member of a group on ``unique``, and whichever member it did not pick lands on ``duplicate``. Pinned
in both directions, because an implementation that ignored ``keep`` for one of the two ports would
still satisfy row conservation and still look right on the port a reader checks first.

**Order is preserved within each port.** Not automatic: the obvious polars formulation
(``DataFrame.unique``) makes no order guarantee unless asked, and the shape that does guarantee it
is a per-row mask (``is_first_distinct`` / ``is_last_distinct``) filtered in place -- which also
happens to be the only shape that can produce both ports from one pass.

**Absent or empty ``columns`` means every column.** The derived-from tool documented a
``["all"]`` sentinel in its notes, its field help and its first example, and never implemented it:
``["all"]`` went straight to the engine, which looked for a column called ``all``. An absent field
is the honest spelling, and a config carried over from the old tool has to be refused by name rather
than quietly searched for.

**A null is equal to a null.** SQL's ``DISTINCT`` and ``GROUP BY`` treat two nulls as one group and
``=`` does not, so a reader of a config cannot guess which one this tool means. It means
``DISTINCT``: the row this tool calls a duplicate is the row a downstream group-by will call a
duplicate, and the alternative would make the tool fail to do its one job exactly on the dirtiest
data, where every null-keyed row would be reported as unique. That choice has to appear in the
tool's own help, so this file checks the help text as well as the behaviour.

Vocabulary is deliberately generic -- code, region, amount -- because the library carries no
caller's nouns.
"""

from collections.abc import Mapping, Sequence
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
from dantron_etl.tools.remove_duplicates import RemoveDuplicates

# Values YAML turns into booleans before a tool ever sees them; no enum in this library may use one.
_YAML_BOOLEANS: Final[frozenset[str]] = frozenset({"true", "false", "yes", "no", "on", "off"})


def _rows() -> pl.DataFrame:
    """One frame carrying every grouping shape worth pinning, identified by row position.

    On all three columns: rows 0 and 6 are identical, rows 3 and 5 are identical *because two nulls
    are one group*, and rows 1, 2 and 4 stand alone. On ``code`` and ``region`` alone: rows 0, 2 and
    6 are one group of three, so the member ``keep`` picks is a real choice rather than a coin flip
    between two. There is deliberately no unique key column -- position is the identity here, which
    lets the full-frame case have genuine full-row duplicates to find.
    """
    return pl.DataFrame(
        {
            "code": ["A", "B", "A", None, "C", None, "A"],
            "region": ["x", "y", "x", None, "z", None, "x"],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 40.0, 10.0],
        }
    )


def _take(frame: pl.DataFrame, positions: Sequence[int]) -> pl.DataFrame:
    """The rows of ``frame`` at ``positions``, in input order and keeping the schema when empty."""
    return frame[list(positions)] if positions else frame.head(0)


def _grouping_cases() -> list[Any]:
    """(frame, config, positions expected on the ``unique`` port) for each grouping worth pinning."""
    rows = _rows()
    return [
        pytest.param(rows, {}, (0, 1, 2, 3, 4), id="absent_columns_means_every_column"),
        pytest.param(rows, {"columns": []}, (0, 1, 2, 3, 4), id="an_empty_column_list_means_every_column"),
        pytest.param(rows, {"keep": "last"}, (1, 2, 4, 5, 6), id="every_column_keeping_the_last_member"),
        pytest.param(rows, {"columns": ["code", "region"]}, (0, 1, 3, 4), id="a_subset_of_columns_defines_sameness"),
        pytest.param(
            rows,
            {"columns": ["code", "region"], "keep": "last"},
            (1, 4, 5, 6),
            id="a_subset_keeping_the_last_member",
        ),
        pytest.param(rows, {"columns": ["code"]}, (0, 1, 3, 4), id="a_single_key_column"),
        pytest.param(
            rows,
            {"columns": ["code", "code"]},
            (0, 1, 3, 4),
            id="a_key_column_named_twice_is_the_same_as_naming_it_once",
        ),
        pytest.param(rows.head(5), {}, (0, 1, 2, 3, 4), id="a_frame_with_no_duplicates_at_all"),
        pytest.param(rows.head(0), {}, (), id="no_rows_at_all"),
    ]


class TestRemoveDuplicatesGrouping:
    """Which member of a group leaves on ``unique``, and that the two ports together are the input."""

    @pytest.mark.parametrize(("frame", "config", "unique_positions"), _grouping_cases())
    def test_every_row_leaves_on_exactly_one_port(
        self,
        frame: pl.DataFrame,
        config: Mapping[str, Any],
        unique_positions: tuple[int, ...],
    ) -> None:
        """Both ports are produced, hold the rows the grouping sent them, and lose nothing between them.

        Comparing each port against a slice of the input pins four things at once: which rows, in
        which order (input order, preserved within each port), in which column order, and with which
        dtypes. The already-unique rows of the no-duplicates case never appear on ``duplicate``, and
        the empty ports of the last two cases still carry the input's schema, so a downstream step
        can still be compiled against them.
        """
        duplicate_positions = tuple(index for index in range(frame.height) if index not in unique_positions)

        result = run_tool(RemoveDuplicates(), Ports({"rows": frame}), config)

        assert result.names == ("unique", "duplicate"), "both ports are declared, unique first"
        assert_frame_equal(result["unique"], _take(frame, unique_positions))
        assert_frame_equal(result["duplicate"], _take(frame, duplicate_positions))
        assert result["unique"].height + result["duplicate"].height == frame.height, "nothing may be dropped"
        assert result["unique"].schema == frame.schema, "a port keeps the input's schema"
        assert result["duplicate"].schema == frame.schema, "a port keeps the input's schema"

    def test_keep_first_and_keep_last_swap_the_members_of_a_group_and_leave_a_lone_row_alone(self) -> None:
        """The two settings are complements: what one sends to ``unique`` the other sends to ``duplicate``.

        This is the pair of facts a single port could not express, and the check an implementation
        that honoured ``keep`` on ``unique`` while always deriving ``duplicate`` from the first
        occurrence would fail -- a defect that leaves row conservation intact and the first port
        correct, so nothing else here would notice it.
        """
        frame = _rows()
        columns = ["code", "region"]

        first = run_tool(RemoveDuplicates(), Ports({"rows": frame}), {"columns": columns})
        last = run_tool(RemoveDuplicates(), Ports({"rows": frame}), {"columns": columns, "keep": "last"})

        assert first["unique"]["amount"].to_list() == [10.0, 20.0, 40.0, 50.0]
        assert first["duplicate"]["amount"].to_list() == [30.0, 40.0, 10.0]
        assert last["unique"]["amount"].to_list() == [20.0, 50.0, 40.0, 10.0]
        assert last["duplicate"]["amount"].to_list() == [10.0, 30.0, 40.0]
        for result in (first, last):
            assert result["unique"].height + result["duplicate"].height == frame.height
            assert result["unique"].height == 4, "one member per group, whichever member that is"

    def test_a_null_key_is_equal_to_another_null_key_and_is_not_an_empty_string(self) -> None:
        """Sameness follows ``DISTINCT``, not ``=``, and the tool's own help has to say so.

        Two rows whose key is missing are one group, so a de-duplication on dirty data actually
        de-duplicates rather than reporting every null-keyed row as unique. A null is still not the
        empty string, which is the shortcut an implementation takes when it builds a text key by
        filling nulls before comparing.
        """
        all_null = pl.DataFrame({"code": [None, None, None], "amount": [1.0, 2.0, 3.0]})

        result = run_tool(RemoveDuplicates(), Ports({"rows": all_null}), {"columns": ["code"]})

        assert result["unique"]["amount"].to_list() == [1.0], "three missing keys are one group"
        assert result["duplicate"]["amount"].to_list() == [2.0, 3.0]

        blank_and_null = pl.DataFrame({"code": [None, ""], "amount": [1.0, 2.0]})

        distinguished = run_tool(RemoveDuplicates(), Ports({"rows": blank_and_null}), {"columns": ["code"]})

        assert distinguished["unique"].height == 2, "a missing value is not an empty string"
        assert distinguished["duplicate"].height == 0

        help_ = RemoveDuplicates().help()
        documented = " ".join((help_.detailed_description, *help_.notes, *help_.field_help.values())).lower()
        assert "null" in documented, "a reader of the config cannot guess this, so it must be written down"

    def test_remove_duplicates_leaves_its_input_untouched(self) -> None:
        """The boundary compares input schemas; the frame itself must be unchanged too."""
        frame = _rows()
        before = frame.clone()

        run_tool(RemoveDuplicates(), Ports({"rows": frame}), {"columns": ["code"], "keep": "last"})

        assert_frame_equal(frame, before)


def _refusal_cases() -> list[Any]:
    """(config, the text its refusal must quote) for each way a key specification can be wrong."""
    return [
        pytest.param({"columns": ["code", "zone"]}, "zone", id="a_key_column_that_is_not_there"),
        pytest.param({"columns": ["Code"]}, "Code", id="a_key_column_whose_case_is_wrong"),
        pytest.param({"columns": ["all"]}, "all", id="the_sentinel_the_derived_from_tool_never_implemented"),
        pytest.param({"columns": [7]}, "columns[0]", id="a_key_column_that_is_not_text"),
    ]


class TestRemoveDuplicatesRefusals:
    """What RemoveDuplicates refuses, and when: off the schema, before a single row is grouped."""

    @pytest.mark.parametrize(("config", "named"), _refusal_cases())
    @pytest.mark.parametrize("height", [7, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_key_column_that_is_not_there_is_refused_by_name(
        self,
        config: Mapping[str, Any],
        named: str,
        height: int,
    ) -> None:
        """``["all"]`` is the case that matters most: it is a real config, from a real older tool.

        Left unrefused it reaches the query engine, which raises about a column named ``all`` that
        the author never wrote and cannot find in their file. Every one of these must be refused in
        this library's own error family, naming the offending entry, and running each against a
        frame with no rows at all shows the refusal is read off the schema rather than off the data.
        """
        frame = _rows().head(height)

        with pytest.raises(ConfigError) as caught:
            run_tool(RemoveDuplicates(), Ports({"rows": frame}), config)

        message = str(caught.value)
        assert named in message, f"refusal should name {named!r}: {message}"

    def test_the_template_declares_an_optional_typed_key_list_and_a_yaml_safe_keep(self) -> None:
        """``columns`` is optional because absent means every column, and both ``keep`` values survive YAML.

        ``first``/``last`` rather than anything resembling a boolean: YAML coerces ``true``,
        ``false``, ``yes``, ``no``, ``on`` and ``off`` before this library sees them, so an enum
        using one of those words is refused with a type error that has nothing to do with what the
        author meant. The old tool's third ``keep`` value is deliberately absent from the enum -- it
        became the ``duplicate`` port.
        """
        template = RemoveDuplicates().config_template()

        assert set(template) == {"columns", "keep"}
        assert template["columns"].type == "array"
        assert template["columns"].required is False, "absent means every column, which is a real answer"
        assert template["columns"].items is not None, "an array with no declared element type checks nothing"
        assert template["columns"].items.type == "string"
        assert template["keep"].default == "first"
        assert set(template["keep"].enum or ()) == {"first", "last"}
        assert not (set(template["keep"].enum or ()) & _YAML_BOOLEANS)
        assert expression_fields(template) == (), "a key column is not a condition; there is no seam here"


class TestRemoveDuplicatesInTheCatalog:
    """Described, documented with examples that actually run, and resolvable by name."""

    def test_remove_duplicates_conforms_and_its_examples_document_both_ports(self) -> None:
        """Conformance runs every example over every declared port, so ``duplicate`` cannot drift.

        That is the port an example is most likely to get wrong and least likely to be read, and it
        is the reason the derived-from tool's own worked example -- which documented a sentinel its
        code did not implement -- would have been caught here before anybody copied it.
        """
        tool = RemoveDuplicates()

        check_tool(tool)

        assert tool.name == "remove_duplicates"
        assert tool.category == "preparation"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("unique", "duplicate")

        examples = tool.help().examples
        assert len(examples) >= 2, "one whole-row de-duplication, and one on named key columns"
        assert any(example.configuration.get("columns") is None for example in examples), (
            "the whole-row case is the common one and needs no columns at all, so an example must show it"
        )
        assert any(example.configuration.get("keep") == "last" for example in examples), (
            "which member survives is the setting most likely to be copied, so it must be run"
        )

    def test_remove_duplicates_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in test_dantron_etl_registry.py."""
        assert "remove_duplicates" in REGISTRY
        assert isinstance(REGISTRY.get("remove_duplicates"), RemoveDuplicates)


class _ReversingRemoveDuplicates(RemoveDuplicates):
    """The tool, with the right rows on the right ports in the wrong order.

    Standing in for the defect this file's third decision exists to catch -- ``DataFrame.unique``
    in place of a per-row mask -- in a form that fails the same way every run rather than however
    the engine happens to order a small frame that day.
    """

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        produced = super().execute(inputs, config)
        return Ports({name: frame.reverse() for name, frame in produced.items()})


class TestTheOrderClaimIsCheckedInTheCatalog:
    """The tool's help says order is preserved within each port; the catalog has to prove it.

    This file already pins order preservation on frames of its own, which is where the defect was
    actually found -- but a reader deciding whether to trust the claim reads the worked example,
    not this file, and conformance compares an example without regard to row order unless the
    example asks. So one of this tool's examples asks, and asking has to be worth something.
    """

    def test_a_worked_example_declares_its_row_order_significant_and_catches_a_reordering(self) -> None:
        """``keep: last`` is the example where a non-order-preserving shortcut hides best.

        It is the configuration a reader is most likely to copy, its ``unique`` port carries two
        distinguishable rows, and the obvious wrong implementation returns exactly those two rows.
        Everything a default comparison looks at is identical; only the sequence differs.
        """
        examples = RemoveDuplicates().help().examples
        keyed = [example for example in examples if example.configuration.get("keep") == "last"]

        assert keyed, "the keep: last example is pinned above; it must still be here"
        assert all(example.check_row_order for example in keyed), (
            "the example demonstrating which member of a group survives must be checked on order, "
            "or the catalog cannot show that order is preserved"
        )

        check_tool(RemoveDuplicates())

        with pytest.raises(ConformanceError) as caught:
            check_tool(_ReversingRemoveDuplicates())

        message = str(caught.value)
        assert "remove_duplicates" in message, f"refusal should name the tool: {message}"
        assert "drifted" in message, f"refusal should say the example no longer matches the tool: {message}"
