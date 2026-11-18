"""Tests for the Formula tool: named columns, each one expression, all read against the input.

Formula takes one input port, ``rows``, and produces one, ``rows``: the input with the columns its
configuration names added or replaced. Its configuration is the first in the library to use the
contract's *nested* shape -- ``columns`` is an array whose element is an object of ``{name, value}``
-- because one step realistically writes several columns and a single ``formula: "[Out] = expr"``
string would put the output name, the expression and the number of columns in one opaque field.

Five rulings are pinned here beyond the happy path, each with a silent failure mode:

**Every expression reads the input frame only.** Column two cannot see column one, even though
column one is listed first. That matches the engine's own multi-column assignment, and the
alternative -- sequential visibility -- makes the result depend on a list order that a configuration
reader cannot see. Pinned by a second entry referencing the first entry's ``name`` and being refused
by the missing column's name.

**Replacement is typed, not incidental.** A step that quietly overwrote a column another step
depends on would be a real hazard: the author of a formula does not necessarily know all forty
columns of the frame they are writing into, so a name collision they never intended must not just
happen. Refusing replacement outright is too strict -- rounding a money column or repairing a code
in place is a legitimate cleanup step -- so the entry says ``replace: true`` and only then is the
column overwritten. Saying ``replace: true`` for a column that does not exist is refused too: that
entry claims to replace something, and a misspelled name would otherwise quietly create a new
column, which is exactly the mistake the flag exists to catch.

**One step writes a column once.** Two entries with the same ``name`` are refused naming it, rather
than the last one silently winning.

**The reportable path reaches inside the array.** A mistake in ``columns[1].value`` is refused by
that path, and ``expression_fields`` / ``expression_values`` report the template path and the
concrete one -- the seam that lets a loader check every expression in a pipeline against the
upstream frame before a single row is read.

**A divisor that is null and a divisor that is zero are two different answers.** The real consumer
is a per-month figure: ``(line_amount / currency_rate) / months_in_period``, a division by the rate
because the rate is quoted the other way round. A null anywhere makes the result null; a zero
divisor gives the engine's infinity, which is *not* a number anyone wants to ship as money. Both are
pinned, because getting either wrong is a wrong figure rather than a failing run, and the way to
refuse an infinity is a rule the author writes (a CaseWhen guard, or an Assert step) rather than a
silent decision made here.

Vocabulary is deliberately generic -- rows, key, code, amount, rate, months -- because the library
carries no caller's nouns.
"""

import math
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
    ToolError,
    check_tool,
    expression_fields,
    expression_values,
    run_tool,
    validate_against_template,
)
from dantron_etl.tools.formula import Formula

# ---------------------------------------------------------------------------
# Test data -- the shape the real derived-column steps run against
# ---------------------------------------------------------------------------


def _rows() -> pl.DataFrame:
    """One frame carrying every shape a real formula step needs.

    A key out of order (so row order is pinned rather than accidentally sorted), a text column with
    a missing value and stray spacing, an amount with a missing value, a rate quoted the other way
    round with a missing value of its own, and a whole-number count of months.
    """
    return pl.DataFrame(
        {
            "key": [4, 1, 3, 2],
            "code": [" a1 ", "b2", None, "c3"],
            "line_amount": [1200.0, 600.0, None, 300.0],
            "currency_rate": [2.0, 1.0, 1.0, None],
            "months_in_period": [12, 6, 3, 3],
        }
    )


def _write_cases() -> list[Any]:
    """(frame, config, written columns, expected column order) for each shape worth pinning."""
    rows = _rows()
    return [
        pytest.param(
            rows,
            {"columns": [{"name": "monthly", "value": "([line_amount] / [currency_rate]) / [months_in_period]"}]},
            {"monthly": [50.0, 100.0, None, None]},
            ("key", "code", "line_amount", "currency_rate", "months_in_period", "monthly"),
            id="one_new_column_appended",
        ),
        pytest.param(
            rows,
            {
                "columns": [
                    {"name": "monthly", "value": "[line_amount] / [months_in_period]"},
                    {"name": "code_key", "value": "upper(strip([code]))"},
                ]
            },
            {
                "monthly": [100.0, 100.0, None, 100.0],
                "code_key": ["A1", "B2", None, "C3"],
            },
            ("key", "code", "line_amount", "currency_rate", "months_in_period", "monthly", "code_key"),
            id="two_new_columns_appended_in_config_order",
        ),
        pytest.param(
            rows,
            {"columns": [{"name": "code", "value": "upper(strip([code]))", "replace": True}]},
            {"code": ["A1", "B2", None, "C3"]},
            ("key", "code", "line_amount", "currency_rate", "months_in_period"),
            id="a_replaced_column_keeps_its_position",
        ),
        pytest.param(
            rows.head(0),
            {"columns": [{"name": "monthly", "value": "[line_amount] / [months_in_period]"}]},
            {"monthly": []},
            ("key", "code", "line_amount", "currency_rate", "months_in_period", "monthly"),
            id="no_rows_at_all",
        ),
    ]


def _same(actual: float | None, expected: float | None) -> bool:
    """Two values that may be null, infinite or not-a-number, compared for equality."""
    if actual is None or expected is None:
        return actual is expected
    if math.isnan(expected):
        return math.isnan(actual)
    return actual == expected


# ---------------------------------------------------------------------------
# What the tool writes, and what it leaves alone
# ---------------------------------------------------------------------------


class TestFormulaWrites:
    """The named columns are written; everything else about the frame survives untouched."""

    @pytest.mark.parametrize(("frame", "config", "written", "expected_columns"), _write_cases())
    def test_every_named_column_is_written_and_every_input_row_and_column_survives(
        self,
        frame: pl.DataFrame,
        config: Mapping[str, Any],
        written: Mapping[str, Sequence[Any]],
        expected_columns: tuple[str, ...],
    ) -> None:
        """Row count, row order and every input column are preserved; only the named columns move.

        Pinning the column order is what says where a new column goes -- appended, in the order the
        list gives -- and that a replaced column stays where it was rather than jumping to the end,
        which would quietly reorder a written table downstream.
        """
        result = run_tool(Formula(), Ports({"rows": frame}), config)

        out = result["rows"]
        assert result.names == ("rows",), "one input port and one output port, both named 'rows'"
        assert out.columns == list(expected_columns)
        assert out.height == frame.height, "a formula step adds columns, never rows"
        assert out["key"].to_list() == frame["key"].to_list(), "row order is the input's"
        for name, values in written.items():
            assert out[name].to_list() == list(values), f"column {name!r}"
        for name in frame.columns:
            if name not in written:
                assert out[name].to_list() == frame[name].to_list(), f"untouched column {name!r}"

    def test_one_step_reads_only_the_input_frame_so_a_later_column_cannot_see_an_earlier_one(self) -> None:
        """The second entry references the first entry's name and is refused by that name.

        Sequential visibility is the tempting alternative and it is the wrong one: it would make the
        result depend on the order of a list that a configuration reader cannot see, and it disagrees
        with the engine's own multi-column assignment. An author who wants the second value to read
        the first writes two Formula steps, which says so.
        """
        frame = _rows()
        config = {
            "columns": [
                {"name": "gross", "value": "[line_amount] * 2"},
                {"name": "net", "value": "[gross] - 1"},
            ]
        }

        with pytest.raises(ToolError) as caught:
            run_tool(Formula(), Ports({"rows": frame}), config)

        message = str(caught.value)
        assert "gross" in message, f"the refusal must name the column that is not there: {message}"
        assert "columns[1].value" in message, f"and the path of the entry that asked for it: {message}"

    @pytest.mark.parametrize(
        ("amount", "rate", "months", "expected"),
        [
            pytest.param(1200.0, 2.0, 12, 50.0, id="the_ordinary_case_divides_by_the_rate"),
            pytest.param(1200.0, None, 12, None, id="no_rate_gives_null_not_zero"),
            pytest.param(1200.0, 2.0, None, None, id="no_month_count_gives_null_not_zero"),
            pytest.param(1200.0, 0.0, 12, math.inf, id="a_zero_rate_gives_the_engines_infinity"),
            pytest.param(1200.0, 2.0, 0, math.inf, id="zero_months_gives_the_engines_infinity"),
            pytest.param(0.0, 0.0, 12, math.nan, id="nothing_over_nothing_is_not_a_number"),
        ],
    )
    def test_the_per_month_figure_divides_by_the_rate_and_the_count_of_months(
        self,
        amount: float,
        rate: float | None,
        months: int | None,
        expected: float | None,
    ) -> None:
        """The money line, and what it does at the edges -- both of which are answers, not failures.

        The rate divides because it is quoted the other way round; a multiplication here would be a
        wrong figure on every row rather than a failing run. A null divisor gives null, which is the
        three-valued answer the rest of the language gives. A zero divisor gives infinity, which is
        the engine's answer and is pinned so it cannot silently become a zero, a null or a raised
        error later: an infinity that must not reach a money column is refused by a rule the author
        writes -- a CaseWhen guard on the rate, or an Assert step -- not by a decision made in here.
        """
        frame = pl.DataFrame(
            {
                "key": [1],
                "line_amount": [amount],
                "currency_rate": pl.Series([rate], dtype=pl.Float64),
                "months_in_period": pl.Series([months], dtype=pl.Int64),
            }
        )
        config = {"columns": [{"name": "monthly", "value": "([line_amount] / [currency_rate]) / [months_in_period]"}]}

        result = run_tool(Formula(), Ports({"rows": frame}), config)

        monthly = result["rows"]["monthly"]
        assert monthly.dtype == pl.Float64
        assert _same(monthly.to_list()[0], expected), f"got {monthly.to_list()[0]!r}, expected {expected!r}"

    def test_an_existing_column_is_overwritten_only_when_the_author_typed_it(self) -> None:
        """Replacement is a hazard worth a keystroke, and the flag is checked in both directions.

        A collision the author never intended -- a name that already exists in a frame they did not
        write -- must be refused rather than silently taken. And ``replace: true`` on a column that
        is *not* there is refused too, because that entry says it is overwriting something: without
        the check, a misspelled name quietly creates a new column and the column the author meant to
        fix keeps its old values.
        """
        frame = _rows()
        collision = {"columns": [{"name": "code", "value": "'x'"}]}

        with pytest.raises(ToolError) as caught:
            run_tool(Formula(), Ports({"rows": frame}), collision)
        assert "code" in str(caught.value), "the refusal names the column that already exists"

        typed = run_tool(
            Formula(), Ports({"rows": frame}), {"columns": [{"name": "code", "value": "'x'", "replace": True}]}
        )
        assert typed["rows"]["code"].to_list() == ["x", "x", "x", "x"]
        assert typed["rows"].columns == frame.columns, "a replaced column keeps its position"

        with pytest.raises(ToolError) as caught:
            run_tool(
                Formula(),
                Ports({"rows": frame}),
                {"columns": [{"name": "cdoe", "value": "'x'", "replace": True}]},
            )
        assert "cdoe" in str(caught.value), "a replacement of a column that is not there names it"


# ---------------------------------------------------------------------------
# Refusals: an author's mistake, in the author's words, before any row is read
# ---------------------------------------------------------------------------


def _malformed_cases() -> list[Any]:
    """(config, the text its refusal must quote) for each mistake in the list itself."""
    return [
        pytest.param({"columns": []}, "columns", id="a_list_with_no_columns_in_it"),
        pytest.param(
            {
                "columns": [
                    {"name": "monthly", "value": "[line_amount] / [months_in_period]"},
                    {"name": "monthly", "value": "[line_amount] * 2"},
                ]
            },
            "monthly",
            id="two_entries_writing_the_same_name",
        ),
        pytest.param(
            {"columns": [{"name": "   ", "value": "[line_amount] * 2"}]},
            "columns[0].name",
            id="an_entry_with_a_blank_name",
        ),
    ]


def _nested_expression_cases() -> list[Any]:
    """(entries, the name its refusal must quote, the path it must report) per schema-only mistake."""
    return [
        pytest.param(
            [{"name": "gross", "value": "[line_amount] * 2"}, {"name": "net", "value": "[amont] - 1"}],
            "amont",
            "columns[1].value",
            id="a_column_that_is_not_there",
        ),
        pytest.param(
            [{"name": "gross", "value": "[line_amount] * 2"}, {"name": "label", "value": "upper([line_amount])"}],
            "line_amount",
            "columns[1].value",
            id="a_text_function_on_a_number",
        ),
        pytest.param(
            [{"name": "flag", "value": "[line_amount] > 'x'"}],
            "line_amount",
            "columns[0].value",
            id="a_comparison_across_type_families",
        ),
    ]


class TestFormulaRefusals:
    """What Formula refuses, and when: on the schema, before a single row is touched."""

    @pytest.mark.parametrize(("config", "quoted"), _malformed_cases())
    def test_a_malformed_list_of_columns_is_refused_naming_the_entry(
        self,
        config: Mapping[str, Any],
        quoted: str,
    ) -> None:
        """An empty list, a repeated name and a blank name are all the tool's own refusals.

        The template cannot express any of them. ``required=True`` treats only a missing value as
        missing, so an empty list passes the template and reaches the tool -- and an empty list is
        the shape a loader produces when the entries were dropped from a file, which must be loud
        rather than a step that quietly writes nothing. A repeated name has to be refused rather
        than letting the last entry silently win, because the reader of the file has no way to know
        which one took effect.
        """
        validate_against_template(Formula().config_template(), config)

        with pytest.raises(ConfigError) as caught:
            run_tool(Formula(), Ports({"rows": _rows()}), config)

        assert quoted in str(caught.value), f"refusal should quote {quoted!r}: {caught.value}"

    @pytest.mark.parametrize(("entries", "named", "path"), _nested_expression_cases())
    @pytest.mark.parametrize("height", [4, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_mistake_in_a_nested_expression_is_refused_before_any_row_is_read(
        self,
        entries: Sequence[Mapping[str, Any]],
        named: str,
        path: str,
        height: int,
    ) -> None:
        """The dtype cases are the point: they are caught only if Formula passes the frame's schema.

        Running each case against a frame with no rows at all is what shows the refusal is read off
        the schema rather than off the data. The path matters as much as the name: a file may hold
        a dozen entries, and ``columns[1].value`` is the only form of address that sends someone to
        the right line.
        """
        frame = _rows().head(height)

        with pytest.raises(ToolError) as caught:
            run_tool(Formula(), Ports({"rows": frame}), {"columns": list(entries)})

        message = str(caught.value)
        assert named in message, f"refusal should name {named!r}: {message}"
        assert path in message, f"refusal should report the path {path!r}: {message}"

    def test_formula_leaves_its_input_untouched(self) -> None:
        """The boundary compares input schemas; the data must be unchanged too."""
        frame = _rows()
        before = frame.clone()

        run_tool(
            Formula(),
            Ports({"rows": frame}),
            {"columns": [{"name": "code", "value": "upper(strip([code]))", "replace": True}]},
        )

        assert_frame_equal(frame, before)


# ---------------------------------------------------------------------------
# The template's nested shape, and the catalog
# ---------------------------------------------------------------------------


class TestFormulaTemplateAndCatalog:
    """The loader seam reaches inside the array, and the tool is described and registered."""

    def test_the_template_declares_an_array_of_named_expressions_and_the_loader_seam_reaches_them(self) -> None:
        """One nested array, and both halves of the seam: the template path and the concrete one.

        This is what lets a pipeline be checked before it runs. A template-only seam would report
        nothing here, because Formula has no top-level expression field at all -- every expression
        it holds sits inside an element of ``columns``.
        """
        template = Formula().config_template()

        assert set(template) == {"columns"}
        assert template["columns"].type == "array"
        assert template["columns"].required is True
        element = template["columns"].items
        assert element is not None, "an untyped array would leave every entry unchecked"
        assert element.type == "object"
        assert set(element.fields or {}) == {"name", "value", "replace"}
        assert (element.fields or {})["name"].required is True
        assert (element.fields or {})["value"].required is True
        assert (element.fields or {})["value"].type == "expression"
        assert (element.fields or {})["replace"].type == "boolean", "a flag YAML can carry as written"

        assert expression_fields(template) == ("columns[].value",), "the template path carries no index"
        config = {
            "columns": [
                {"name": "monthly", "value": "[line_amount] / [months_in_period]"},
                {"name": "code_key", "value": "upper([code])"},
            ]
        }
        assert expression_values(template, config) == (
            ("columns[0].value", "[line_amount] / [months_in_period]"),
            ("columns[1].value", "upper([code])"),
        )

    def test_formula_conforms_and_its_examples_show_the_replacement_flag(self) -> None:
        """Conformance runs every worked example, so a documented output is a checked output."""
        tool = Formula()

        check_tool(tool)

        assert tool.name == "formula"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",)

        examples = tool.help().examples
        assert len(examples) >= 2, "one deriving a column, and one showing a deliberate overwrite"
        assert any(entry.get("replace") for example in examples for entry in example.configuration["columns"]), (
            "replacement is the behaviour most likely to surprise, so an example must show it"
        )
        assert any(len(example.configuration["columns"]) > 1 for example in examples), (
            "writing several columns in one step is the reason the config is a list, so show it"
        )

    def test_formula_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in tests/core/test_registry.py."""
        assert "formula" in REGISTRY
        assert isinstance(REGISTRY.get("formula"), Formula)


# ---------------------------------------------------------------------------
# The tool that already had the answer keeps it
#
# Formula is one of four tools that refused a blank name before any shared rule existed, and it
# refuses one in its own words, from its own ``validate_config``, at the configuration phase --
# before the frames are even looked at. A shared rule placed in the contract's seam runs later, at
# ``validate_inputs``, so it can never be what answers here.
#
# That makes Formula's own check redundant in the sense that deleting it would still leave a blank
# refused. It is pinned rather than deleted: the earlier refusal is the better one -- it does not
# need a frame to be right -- and a change that quietly moved this wording would be a change to a
# tool that was never broken.
# ---------------------------------------------------------------------------


class TestFormulaStillRefusesABlankNameInItsOwnWords:
    """Unchanged by the shared rule, in every shape of blank and at the same phase as before."""

    @pytest.mark.parametrize("name", ["", " ", "   ", "\t", "\n"])
    def test_a_blank_name_is_refused_by_formula_itself(self, name: str) -> None:
        """The exact sentence it gave before, quoting the entry by its path in the list."""
        with pytest.raises(ConfigError) as caught:
            run_tool(Formula(), Ports({"rows": _rows()}), {"columns": [{"name": name, "value": "[line_amount] * 2"}]})

        assert str(caught.value) == "config field 'columns[0].name' must not be blank", (
            f"Formula was never broken here, and its wording must not move: {caught.value}"
        )

    def test_the_entry_that_is_blank_is_the_one_named(self) -> None:
        """Three entries, one blank, and the refusal has to say which -- a list has no other address."""
        with pytest.raises(ConfigError) as caught:
            run_tool(
                Formula(),
                Ports({"rows": _rows()}),
                {
                    "columns": [
                        {"name": "gross", "value": "[line_amount] * 2"},
                        {"name": "   ", "value": "[line_amount] + 1"},
                        {"name": "net", "value": "[line_amount] - 1"},
                    ]
                },
            )

        assert "columns[1].name" in str(caught.value), f"the blank entry is the second one: {caught.value}"
