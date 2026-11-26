"""A '/' whose divisor is zero stops the step, naming the field, the expression and the rows.

The owner's ruling: a division by zero has no answer, so the step refuses rather than write
infinity or not-a-number, which a later filter reads as larger than every number. A flow that
expects zeros guards them with case_when: a rung that sends the zero rows elsewhere comes first,
and the rung that divides then never sees them. So case_when checks each division only on the
rows that reach it, as SQL's CASE evaluates a branch only for the rows that take it.

A null divisor is not zero: the division gives null, as every other operation on a null does.
0.0 and -0.0 are zero. NaN is not zero: dividing by it gives NaN, as it always has.
"""

from typing import Any

import polars as pl
import pytest

from dantron_etl.authoring import REGISTRY, DataError, Ports, run_tool
from dantron_etl.config import expression_fields
from dantron_etl.expr import ExpressionError, compile_expr

ROWS = pl.DataFrame(
    {
        "key": ["a", "b", "c", "d", "e", "f", "g"],
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        "months": [2, 0, 5, 0, None, 0, 0],
        "text": ["x", "y", "z", "w", "v", "u", "t"],
    }
)

# One configuration per tool that reads an expression, each dividing by [months] in one field.
DIVIDING: dict[str, tuple[dict[str, Any], str]] = {
    "formula": ({"columns": [{"name": "each", "value": "[amount] / [months]"}]}, "columns[0].value"),
    "filter": ({"predicate": "[amount] / [months] > 1"}, "predicate"),
    "assert_rows": ({"predicate": "coalesce([amount] / [months], 99.0) > 1"}, "predicate"),
    "case_when": (
        {"cases": [{"when": "[amount] > 0", "then": "[amount] / [months]"}], "otherwise": "0.0", "output": "each"},
        "cases[0].then",
    ),
    "group_by": (
        {"by": ["key"], "aggregations": [{"name": "each", "function": "sum", "value": "[amount] / [months]"}]},
        "aggregations[0].value",
    ),
    "assemble_date": (
        {"year": "2026", "month": "1", "day": "format_number([amount] / [months], 0, '', '.')", "output": "day"},
        "day",
    ),
    "normalize_text": (
        {"source": "format_number([amount] / [months], 2, ',', '.')", "output": "clean", "operations": ["lower"]},
        "source",
    ),
    "multi_row_formula": ({"output": "each", "expression": "[amount] / [months]"}, "expression"),
    "regex_extract_groups": (
        {"source": "format_number([amount] / [months], 2, ',', '.')", "pattern": "(?P<digits>\\d+)"},
        "source",
    ),
}


# No expression these two accept can divide: '/' always gives a decimal, and nothing in the language
# turns a decimal into a date (add_months and make_date take whole numbers). They still check.
CANNOT_DIVIDE: frozenset[str] = frozenset({"date_diff", "expand_date_range"})


def tools_with_expressions() -> set[str]:
    return {name for name in REGISTRY.names() if expression_fields(REGISTRY.get(name).config_template())}


def test_every_tool_that_reads_an_expression_is_checked_here() -> None:
    """A tool added with an expression field has to be added above, or this fails."""
    assert set(DIVIDING) | CANNOT_DIVIDE == tools_with_expressions()
    assert not set(DIVIDING) & CANNOT_DIVIDE


@pytest.mark.parametrize("tool", sorted(DIVIDING))
def test_a_zero_divisor_stops_the_step_naming_the_field_the_expression_and_the_rows(tool: str) -> None:
    config, field = DIVIDING[tool]
    with pytest.raises(DataError) as caught:
        run_tool(REGISTRY.get(tool), Ports({"rows": ROWS}), config)

    message = str(caught.value)
    assert field in message, message
    assert "[months]" in message and "zero" in message, message
    assert "4 of 7 row(s)" in message, message
    assert "'key': 'b'" in message, message
    assert caught.value.offending_rows == 4


@pytest.mark.parametrize("tool", sorted(DIVIDING))
def test_the_same_step_runs_when_no_divisor_is_zero(tool: str) -> None:
    config, _ = DIVIDING[tool]
    rows = ROWS.filter(pl.col("months").is_null() | (pl.col("months") != 0))
    run_tool(REGISTRY.get(tool), Ports({"rows": rows}), config)


@pytest.mark.parametrize("tool", sorted(DIVIDING))
def test_an_empty_frame_divides_nothing_and_is_not_stopped(tool: str) -> None:
    config, _ = DIVIDING[tool]
    if tool == "assert_rows":
        config = {**config, "allow_empty": True}
    run_tool(REGISTRY.get(tool), Ports({"rows": ROWS.head(0)}), config)


def formula(rows: pl.DataFrame, value: str) -> list[Any]:
    config = {"columns": [{"name": "out", "value": value}]}
    return run_tool(REGISTRY.get("formula"), Ports({"rows": rows}), config)["rows"]["out"].to_list()


class TestWhatCountsAsZero:
    def test_a_null_divisor_gives_null_and_is_not_stopped(self) -> None:
        rows = pl.DataFrame({"a": [1.0, 2.0], "b": [None, 4.0]})
        assert formula(rows, "[a] / [b]") == [None, 0.5]

    @pytest.mark.parametrize("zero", [0.0, -0.0])
    def test_a_decimal_zero_of_either_sign_is_zero(self, zero: float) -> None:
        rows = pl.DataFrame({"a": [1.0], "b": [zero]})
        with pytest.raises(DataError):
            formula(rows, "[a] / [b]")

    def test_a_whole_number_zero_is_zero(self) -> None:
        rows = pl.DataFrame({"a": [1], "b": [0]})
        with pytest.raises(DataError):
            formula(rows, "[a] / [b]")

    def test_a_not_a_number_divisor_is_not_zero_and_gives_not_a_number(self) -> None:
        rows = pl.DataFrame({"a": [1.0], "b": [float("nan")]})
        (out,) = formula(rows, "[a] / [b]")
        assert out != out

    def test_a_zero_numerator_over_a_divisor_is_fine(self) -> None:
        rows = pl.DataFrame({"a": [0.0], "b": [4.0]})
        assert formula(rows, "[a] / [b]") == [0.0]

    def test_a_divisor_built_from_an_expression_is_checked_as_a_whole(self) -> None:
        rows = pl.DataFrame({"a": [1.0, 1.0], "b": [3.0, 2.0]})
        with pytest.raises(DataError) as caught:
            formula(rows, "[a] / ([b] - 2)")
        assert "([b] - 2)" in str(caught.value) or "[b] - 2" in str(caught.value)

    def test_the_second_division_in_one_expression_is_checked_too(self) -> None:
        rows = pl.DataFrame({"a": [1.0], "b": [2.0], "c": [0.0]})
        with pytest.raises(DataError) as caught:
            formula(rows, "[a] / [b] + [a] / [c]")
        assert "[c]" in str(caught.value)

    def test_a_literal_zero_divisor_is_refused_before_any_row_is_read(self) -> None:
        with pytest.raises(ExpressionError) as caught:
            compile_expr("[a] / 0", {"a": pl.Float64})
        assert "zero" in str(caught.value)
        with pytest.raises(ExpressionError):
            compile_expr("[a] / 0.0")
        with pytest.raises(ExpressionError):
            compile_expr("[a] / -0.0")


class TestAShortList:
    def test_no_more_than_five_rows_are_shown_and_every_one_is_counted(self) -> None:
        rows = pl.DataFrame({"n": list(range(40)), "a": [1.0] * 40, "b": [0.0] * 40})
        with pytest.raises(DataError) as caught:
            formula(rows, "[a] / [b]")
        assert caught.value.offending_rows == 40 and len(caught.value.examples) == 5
        assert "40 of 40 row(s)" in str(caught.value)


def case_when(rows: pl.DataFrame, cases: list[dict[str, str]], otherwise: str | None = None) -> list[Any]:
    config: dict[str, Any] = {"cases": cases, "output": "out"}
    if otherwise is not None:
        config["otherwise"] = otherwise
    return run_tool(REGISTRY.get("case_when"), Ports({"rows": rows}), config)["rows"]["out"].to_list()


class TestCaseWhenChecksADivisionOnlyWhereItIsUsed:
    rows = pl.DataFrame({"a": [10.0, 20.0, 30.0], "b": [2.0, 0.0, None]})

    def test_a_rung_that_takes_the_zero_rows_first_guards_the_rung_after_it(self) -> None:
        out = case_when(self.rows, [{"when": "[b] = 0", "then": "null"}, {"when": "true", "then": "[a] / [b]"}])
        assert out == [5.0, None, None]

    def test_the_guard_works_the_other_way_round_in_otherwise(self) -> None:
        out = case_when(self.rows, [{"when": "[b] = 0 or [b] is null", "then": "0.0"}], otherwise="[a] / [b]")
        assert out == [5.0, 0.0, 0.0]

    def test_a_rung_whose_own_condition_lets_a_zero_through_is_stopped(self) -> None:
        with pytest.raises(DataError) as caught:
            case_when(self.rows, [{"when": "[a] > 0", "then": "[a] / [b]"}])
        assert "cases[0].then" in str(caught.value) and "1 of 3 row(s)" in str(caught.value)

    def test_otherwise_is_stopped_on_a_zero_row_that_reaches_it(self) -> None:
        with pytest.raises(DataError) as caught:
            case_when(self.rows, [{"when": "[a] > 15.0", "then": "0.0"}], otherwise="[a] / ([b] - 2)")
        assert "otherwise" in str(caught.value) and "1 of 3 row(s)" in str(caught.value)

    def test_a_condition_is_checked_only_on_the_rows_no_earlier_rung_took(self) -> None:
        out = case_when(
            self.rows,
            [{"when": "[b] = 0", "then": "'none'"}, {"when": "[a] / [b] > 4", "then": "'big'"}],
            otherwise="'small'",
        )
        assert out == ["big", "none", "small"]

    def test_a_condition_reached_by_a_zero_row_is_stopped(self) -> None:
        with pytest.raises(DataError) as caught:
            case_when(self.rows, [{"when": "[a] / [b] > 4", "then": "'big'"}], otherwise="'small'")
        assert "cases[0].when" in str(caught.value)

    def test_a_null_condition_falls_through_to_the_next_rung_for_the_check_as_for_the_value(self) -> None:
        rows = pl.DataFrame({"a": [10.0], "b": [0.0], "flag": [None]}, schema={"a": pl.Float64, "b": pl.Float64, "flag": pl.Boolean})
        with pytest.raises(DataError):
            case_when(rows, [{"when": "[flag]", "then": "0.0"}], otherwise="[a] / [b]")
