"""Tests for the Filter tool: one input, two outputs, and nothing dropped in between.

Filter takes one input port, ``rows``, and produces two: ``true`` and ``false``, named after the
value the predicate took. Every row leaves on exactly one of them, and that is the whole reason
the tool has two outputs rather than one -- the rows a filter discards stop being a number a reader
reconstructs afterwards with an anti-join and become a port a flow can route to a report or to an
assertion.

Three decisions are pinned here beyond the happy path, because each has a silent failure mode:

**Row conservation.** ``true.height + false.height == rows.height`` for every case, nulls included.
That is the invariant that makes "nothing is dropped" a fact rather than a claim, and it is checked
by comparing both ports against slices of the input, so column order, row order and dtypes are
pinned by the same assertion.

**Null routing.** A predicate is three-valued: a comparison against a missing value is null, not
false. A null row leaves on ``false`` by default -- matching the engine, so it surprises nobody --
and ``nulls: "true"`` sends it to ``true`` instead. There is deliberately no third port: the
distinction is recoverable downstream in one step (``[type] is null`` on the ``false`` port), and a
third output would burden every flow with an edge that is almost always empty. What must not be
lost is that a null is *not* false, so the test below moves the null row without moving the row
that is genuinely false.

**A schema, not a list of column names.** Filter holds the frame, so it can hand the compiler the
frame's *schema*, and the compiler can then refuse ``upper([amount])`` on a number in the author's
words rather than letting the query engine raise about a dtype three layers down. A list of column
names would catch a misspelled column and nothing else. The refusals below run against a frame with
no rows at all, which is what proves the refusal is read off the schema before any row is touched.

Vocabulary is deliberately generic -- rows, type, disposition, amount, code, status -- because the
library carries no caller's nouns; these are the shapes of three real filter steps, not their names.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.expr import compile_expr
from dantron_etl.tool import (
    CATEGORIES,
    REGISTRY,
    ConfigError,
    Ports,
    ToolError,
    check_tool,
    compiled_dtype,
    expression_fields,
    run_tool,
)
from dantron_etl.tools.filter import Filter

# ---------------------------------------------------------------------------
# Test data -- the shape the real filter steps run against
# ---------------------------------------------------------------------------


def _rows() -> pl.DataFrame:
    """One frame carrying every shape the three real filter steps need.

    A text column with a missing value (``type``), a second text column selected on a fixed value
    (``disposition``), a truth column with a missing value (``posted``), a number, and a text
    column whose missing value makes a text function return null rather than false.
    """
    return pl.DataFrame(
        {
            "key": [3, 1, 4, 2],
            "type": [None, "receivable", "receivable", "payable"],
            "disposition": ["ignored", "counted", None, "counted"],
            "posted": [True, True, None, False],
            "amount": [0.0, 100.0, 25.0, -50.0],
            "code": ["X3", "X1", None, "Y2"],
        }
    )


def _take(frame: pl.DataFrame, positions: Sequence[int]) -> pl.DataFrame:
    """The rows of ``frame`` at ``positions``, in input order and keeping the schema when empty."""
    return frame[list(positions)] if positions else frame.head(0)


def _routing_cases() -> list[Any]:
    """(frame, config, positions expected on the ``true`` port) for each routing case worth pinning."""
    rows = _rows()
    return [
        pytest.param(
            rows,
            {"predicate": "[type] = 'receivable'"},
            (1, 2),
            id="equality_on_a_type_column_a_null_type_passes_no_filter",
        ),
        pytest.param(
            rows,
            {"predicate": "[disposition] = 'counted'"},
            (1, 3),
            id="equality_selects_the_rows_carrying_a_figure",
        ),
        pytest.param(rows, {"predicate": "[posted] = true"}, (0, 1), id="a_posted_status_filter"),
        pytest.param(
            rows,
            {"predicate": "[amount] > 0 and starts_with([code], 'X')"},
            (1,),
            id="two_conditions_and_a_text_function",
        ),
        pytest.param(rows, {"predicate": "[key] > 0"}, (0, 1, 2, 3), id="every_row_matches"),
        pytest.param(rows, {"predicate": "[key] < 0"}, (), id="no_row_matches"),
        pytest.param(rows.head(0), {"predicate": "[type] = 'receivable'"}, (), id="no_rows_at_all"),
    ]


# ---------------------------------------------------------------------------
# Routing: what each port holds, and that between them they hold everything
# ---------------------------------------------------------------------------


class TestFilterRouting:
    """The predicate decides a port per row, and the two ports together are the input."""

    @pytest.mark.parametrize(("frame", "config", "true_positions"), _routing_cases())
    def test_every_row_leaves_on_exactly_one_port(
        self,
        frame: pl.DataFrame,
        config: Mapping[str, Any],
        true_positions: tuple[int, ...],
    ) -> None:
        """Both ports are produced, hold the rows the predicate sent them, and lose nothing.

        Comparing each port against a slice of the input pins four things at once: which rows, in
        which order (input order is preserved -- polars filters in order and Filter does not
        reorder), in which column order, and with which dtypes.
        """
        false_positions = tuple(index for index in range(frame.height) if index not in true_positions)

        result = run_tool(Filter(), Ports({"rows": frame}), config)

        assert result.names == ("true", "false"), "both ports are declared, true first"
        assert_frame_equal(result["true"], _take(frame, true_positions))
        assert_frame_equal(result["false"], _take(frame, false_positions))
        assert result["true"].height + result["false"].height == frame.height, "nothing may be dropped"
        assert result["true"].schema == frame.schema, "a port keeps the input's schema"
        assert result["false"].schema == frame.schema, "a port keeps the input's schema"

    def test_a_null_predicate_is_routed_by_the_nulls_setting_and_is_not_the_same_as_false(self) -> None:
        """A null row goes to false by default and to true on request; a false row never moves.

        This is the pair of facts a single port could not express. If a null were simply treated as
        false, the two runs below would be identical, and the 'could not tell' rows would be
        indistinguishable from the rows that were tested and failed.
        """
        frame = pl.DataFrame({"key": [7, 5, 6], "type": ["receivable", "payable", None]})
        predicate = "[type] = 'receivable'"

        default = run_tool(Filter(), Ports({"rows": frame}), {"predicate": predicate})
        requested = run_tool(Filter(), Ports({"rows": frame}), {"predicate": predicate, "nulls": "include"})

        assert default["true"]["key"].to_list() == [7]
        assert default["false"]["key"].to_list() == [5, 6], "a null predicate goes to false by default"
        assert requested["true"]["key"].to_list() == [7, 6], "nulls: 'true' moves the null row"
        assert requested["false"]["key"].to_list() == [5], "the row that is genuinely false does not move"
        for result in (default, requested):
            assert result["true"].height + result["false"].height == frame.height

    def test_an_empty_port_still_carries_the_input_schema(self) -> None:
        """A port nothing reached is empty but typed, so a downstream predicate finds its columns."""
        frame = _rows()

        result = run_tool(Filter(), Ports({"rows": frame}), {"predicate": "[type] = 'no_such_value'"})

        assert result["true"].height == 0
        assert result["true"].columns == frame.columns
        assert result["true"].schema == frame.schema
        # The reason it matters: the next step compiles its own predicate against this port.
        compile_expr("[amount] > 0", result["true"].schema)

    def test_a_bare_truth_column_is_a_predicate_but_a_number_is_not(self) -> None:
        """A boolean column filters as written; a number is refused here, not by the query engine.

        ``[amount]`` is the slip an author makes when they meant ``[amount] > 0``. Handed to the
        engine it raises about a dtype, from inside a library the author has never heard of; the
        refusal has to name the column and belong to this library's error family.
        """
        frame = _rows()

        result = run_tool(Filter(), Ports({"rows": frame}), {"predicate": "[posted]"})

        assert result["true"]["key"].to_list() == [3, 1], "in input order, which Filter preserves"
        assert result["false"]["key"].to_list() == [4, 2], "a null status is not true"

        with pytest.raises(ToolError) as caught:
            run_tool(Filter(), Ports({"rows": frame}), {"predicate": "[amount]"})
        assert "amount" in str(caught.value)

    def test_filter_leaves_its_input_untouched(self) -> None:
        """The boundary compares input schemas; the data must be unchanged too."""
        frame = _rows()
        before = frame.clone()

        run_tool(Filter(), Ports({"rows": frame}), {"predicate": "[disposition] = 'counted'"})

        assert_frame_equal(frame, before)


# ---------------------------------------------------------------------------
# Refusals: an author's mistake, in the author's words, before any row is read
# ---------------------------------------------------------------------------


def _refusal_cases() -> list[Any]:
    """(predicate, the name its refusal must quote) for each mistake only a schema can catch."""
    return [
        pytest.param("[typ] = 'receivable'", "typ", id="a_column_that_is_not_there"),
        pytest.param("upper([amount]) = 'X'", "amount", id="a_text_function_on_a_number"),
        pytest.param("[amount] > 'x'", "amount", id="a_comparison_across_type_families"),
    ]


class TestFilterRefusals:
    """What Filter refuses, and when: on the schema, before a single row is touched."""

    @pytest.mark.parametrize(("predicate", "named"), _refusal_cases())
    @pytest.mark.parametrize("height", [4, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_predicate_the_schema_refuses_is_refused_before_any_row_is_read(
        self,
        predicate: str,
        named: str,
        height: int,
    ) -> None:
        """The dtype cases are the point: they are caught only if Filter passes the frame's schema.

        A list of column names would catch the misspelled column and let the other two through, to
        fail later inside the query engine. Running each case against a frame with no rows at all
        is what shows the refusal comes from the schema rather than from the data, and the message
        has to quote the offending name and the predicate it sits in, because these are read in a
        log by someone holding a file of many predicates.
        """
        frame = _rows().head(height)

        with pytest.raises(ToolError) as caught:
            run_tool(Filter(), Ports({"rows": frame}), {"predicate": predicate})

        message = str(caught.value)
        assert named in message, f"refusal should name {named!r}: {message}"
        assert predicate in message, f"refusal should quote the predicate: {message}"

    def test_an_unresolved_parameter_placeholder_reaches_filter_and_is_refused(self) -> None:
        """A predicate is authored with a placeholder resolved from a rules file before Filter runs.

        One that arrives unresolved must be refused loudly: passed through, it compiles as ordinary
        text, matches nothing, and leaves the run looking as if it had succeeded.
        """
        with pytest.raises(ConfigError) as caught:
            run_tool(Filter(), Ports({"rows": _rows()}), {"predicate": "[type] = '${rules.wanted_type}'"})

        assert "${rules.wanted_type}" in str(caught.value)

    def test_the_template_declares_one_expression_predicate_and_routes_nulls_to_false_by_default(self) -> None:
        """The predicate is the whole condition -- no column/operator/value structure beside it."""
        template = Filter().config_template()

        assert set(template) == {"predicate", "nulls"}
        assert expression_fields(template) == ("predicate",), "the loader seam is the declared type"
        assert template["predicate"].required is True
        assert template["nulls"].default == "exclude"
        assert set(template["nulls"].enum or ()) == {"exclude", "include"}


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


class TestFilterInTheCatalog:
    """Filter is described, documented with examples that actually run, and resolvable by name."""

    def test_filter_conforms_and_its_examples_document_null_routing(self) -> None:
        """Conformance runs every worked example, so a documented output is a checked output."""
        tool = Filter()

        check_tool(tool)

        assert tool.name == "filter"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("true", "false")

        examples = tool.help().examples
        assert len(examples) >= 2, "one plain example, and one showing where a null row goes"
        assert any("nulls" in example.configuration for example in examples), (
            "null routing is the behaviour most likely to surprise, so an example must show it"
        )

    def test_filter_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in tests/core/test_registry.py."""
        assert "filter" in REGISTRY
        assert isinstance(REGISTRY.get("filter"), Filter)


# ---------------------------------------------------------------------------
# The predicate check, moved into the contract
#
# Filter hand-rolled "the result must be a truth value" -- a select against the schema, the first
# entry of it, and a message of its own. Three more tools need the same check and a fourth needs
# the probe underneath it, so the check moves to ``compile_predicate_for`` beside ``compile_for``.
# What must not change is what Filter refuses and what it says, which is what this pins.
# ---------------------------------------------------------------------------


class TestFilterRefusesANonPredicateThroughTheContract:
    """Filter's refusal survives the move, word for word in the parts a reader depends on."""

    @pytest.mark.parametrize(
        ("predicate", "named"),
        [
            pytest.param("[amount]", "amount", id="a_number_is_not_a_truth_value"),
            pytest.param("[code]", "code", id="text_is_not_a_truth_value"),
            pytest.param("upper([code])", "code", id="a_text_function_is_not_a_truth_value"),
        ],
    )
    @pytest.mark.parametrize("height", [4, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_predicate_that_is_not_a_condition_is_refused_by_the_shared_compiler(
        self,
        predicate: str,
        named: str,
        height: int,
    ) -> None:
        """The same refusal as before: the column, the config field, the text, and the dtype.

        Running it against a frame with no rows at all is what shows the answer is read off the
        schema and not off the data -- and the shared version has to keep that, because the tool
        that will reuse it refuses a whole run on the strength of it.
        """
        frame = _rows().head(height)

        with pytest.raises(ConfigError) as caught:
            run_tool(Filter(), Ports({"rows": frame}), {"predicate": predicate})

        message = str(caught.value)
        assert named in message, f"the refusal must name the column: {message}"
        assert "predicate" in message, f"the refusal must name the config field: {message}"
        assert predicate in message, f"the refusal must quote the expression: {message}"
        assert str(compiled_dtype(frame, predicate, "predicate")[1]) in message, (
            f"the refusal must say what the expression produced instead: {message}"
        )
