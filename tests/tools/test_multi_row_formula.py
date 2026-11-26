"""Tests for the MultiRowFormula tool: a column set by an expression that can read other rows.

``[Row-1:amount]`` is ``amount`` one row up and ``[Row+2:amount]`` two rows down, in the order the
rows arrived, optionally within groups. A row that is not there (above the first, below the last,
or in another group) reads as null. ``[Row-N:<output>]`` reads what this step already wrote for
that earlier row, which is how fill-down and running balances work; ``[Row+N:<output>]`` is refused
because that value is not written yet. A plain ``[<output>]`` on an existing column is the current
row's original value.

The tool is reached only through ``get_tool`` and ``run_tool``, so these tests do not presume its
module or class. Vocabulary is generic: accounts, orders, amounts, balances.
"""

import math
import threading
import time
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl import ColumnCheckError, PortRef, get_tool, load_pipeline, run_pipeline
from dantron_etl.authoring import CATEGORIES, ConfigError, DataError, Ports, check_tool, run_tool

TOOL_NAME = "multi_row_formula"


def _run(frame: pl.DataFrame, config: Mapping[str, Any]) -> pl.DataFrame:
    """Run the tool on one frame and return its ``rows`` output."""
    return run_tool(get_tool(TOOL_NAME), Ports({"rows": frame}), dict(config))["rows"]


def _orders() -> pl.DataFrame:
    """Four orders in no sorted order, so arrival order is what the references must follow."""
    return pl.DataFrame(
        {
            "order_id": [4, 1, 3, 2],
            "amount": [10.0, 20.0, 5.0, 40.0],
            "quantity": [1.0, 2.0, 4.0, 8.0],
        }
    )


# ---------------------------------------------------------------------------
# Reading other rows
# ---------------------------------------------------------------------------


class TestRowReferences:
    """``[Row-N:x]`` and ``[Row+N:x]`` read N rows up or down; past either end they read null."""

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            pytest.param("[Row-1:amount]", [None, 10.0, 20.0, 5.0], id="one_row_up_first_row_reads_null"),
            pytest.param("[Row+1:amount]", [20.0, 5.0, 40.0, None], id="one_row_down_last_row_reads_null"),
            pytest.param("[Row-2:amount]", [None, None, 10.0, 20.0], id="two_rows_up"),
            pytest.param("[Row+2:amount]", [5.0, 40.0, None, None], id="two_rows_down"),
            pytest.param("[amount] - [Row-1:amount]", [None, 10.0, -15.0, 35.0], id="change_from_the_row_above"),
            pytest.param(
                "coalesce([Row-1:amount], 0)", [0.0, 10.0, 20.0, 5.0], id="coalesce_turns_the_missing_row_into_zero"
            ),
        ],
    )
    def test_a_new_column_is_appended_and_reads_the_rows_around_it(
        self, expression: str, expected: Sequence[float | None]
    ) -> None:
        frame = _orders()

        out = _run(frame, {"output": "result", "expression": expression})

        assert out.columns == [*frame.columns, "result"], "a new column goes on the end"
        assert out["result"].to_list() == list(expected)
        assert_frame_equal(out.drop("result"), frame, check_dtypes=True)

    def test_an_existing_column_is_updated_in_place_and_keeps_its_position(self) -> None:
        """Mirror of the appended case: the column stays where it was and nothing else moves."""
        frame = _orders()

        out = _run(frame, {"output": "amount", "expression": "[amount] * coalesce([Row+1:quantity], 0)"})

        assert out.columns == frame.columns
        assert out["amount"].to_list() == [20.0, 80.0, 40.0, 0.0]
        assert out["order_id"].to_list() == [4, 1, 3, 2], "row order is the input's"
        assert out["quantity"].to_list() == frame["quantity"].to_list()

    def test_the_furthest_reach_is_a_thousand_rows_in_either_direction(self) -> None:
        """1000 is accepted and reaches exactly 1000 rows; 0 and 1001 are refused in the next class."""
        frame = pl.DataFrame({"amount": [float(i) for i in range(1001)]})

        back = _run(frame, {"output": "back", "expression": "[Row-1000:amount]"})["back"].to_list()
        ahead = _run(frame, {"output": "ahead", "expression": "[Row+1000:amount]"})["ahead"].to_list()

        assert back == [None] * 1000 + [0.0]
        assert ahead == [1000.0] + [None] * 1000

    @pytest.mark.parametrize(
        "expression",
        [
            pytest.param("[Row - 1 : unit price]", id="spaces_around_the_sign_and_the_colon"),
            pytest.param("[Row-1:unit price]", id="a_column_name_with_a_space_in_it"),
        ],
    )
    def test_spaces_inside_the_brackets_are_allowed(self, expression: str) -> None:
        frame = pl.DataFrame({"unit price": [3.0, 4.0, 5.0]})

        out = _run(frame, {"output": "previous_price", "expression": expression})

        assert out["previous_price"].to_list() == [None, 3.0, 4.0]


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


class TestGroups:
    """Within ``group_by``, each group is read in arrival order and references never cross into another."""

    def _interleaved(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "account": ["A", "B", "A", "B", "A"],
                "amount": [1.0, 10.0, 2.0, 20.0, 3.0],
            }
        )

    @pytest.mark.parametrize(
        ("expression", "group_by", "expected"),
        [
            pytest.param(
                "[amount] - [Row-1:amount]", ["account"], [None, None, 1.0, 10.0, 1.0], id="change_within_each_account"
            ),
            pytest.param(
                "[amount] - [Row-1:amount]", None, [None, 9.0, -8.0, 18.0, -17.0], id="without_groups_rows_are_one_run"
            ),
            pytest.param("[Row+1:amount]", ["account"], [2.0, 20.0, 3.0, None, None], id="next_within_each_account"),
            pytest.param("[Row+2:amount]", ["account"], [3.0, None, None, None, None], id="two_down_within_an_account"),
        ],
    )
    def test_interleaved_groups_are_each_read_in_arrival_order(
        self, expression: str, group_by: list[str] | None, expected: Sequence[float | None]
    ) -> None:
        frame = self._interleaved()
        config: dict[str, Any] = {"output": "result", "expression": expression}
        if group_by is not None:
            config["group_by"] = group_by

        out = _run(frame, config)

        assert out["result"].to_list() == list(expected)
        assert out["account"].to_list() == frame["account"].to_list(), "rows are not regrouped or sorted"

    def test_a_null_group_value_is_a_group_of_its_own(self) -> None:
        """As in SQL's PARTITION BY: the rows with no account form one group, apart from account A."""
        frame = pl.DataFrame({"account": ["A", None, "A", None], "amount": [1.0, 5.0, 2.0, 7.0]})

        out = _run(frame, {"output": "previous", "expression": "[Row-1:amount]", "group_by": ["account"]})

        assert out["previous"].to_list() == [None, None, 1.0, 5.0]


# ---------------------------------------------------------------------------
# Self-reference: reading what this step already wrote
# ---------------------------------------------------------------------------


class TestSelfReference:
    """``[Row-N:<output>]`` reads the value written for that earlier row, not the input's value."""

    @pytest.mark.parametrize(
        ("amounts", "expected"),
        [
            pytest.param(
                [5.0, None, None, None, 8.0, None],
                [5.0, 5.0, 5.0, 5.0, 8.0, 8.0],
                id="a_run_of_several_blanks_is_filled_from_the_last_value_above",
            ),
            pytest.param(
                [None, None, 3.0, None],
                [None, None, 3.0, 3.0],
                id="a_blank_at_the_very_top_stays_blank",
            ),
        ],
    )
    def test_fill_down(self, amounts: Sequence[float | None], expected: Sequence[float | None]) -> None:
        frame = pl.DataFrame({"amount": pl.Series(amounts, dtype=pl.Float64)})

        out = _run(frame, {"output": "amount", "expression": "coalesce([amount], [Row-1:amount])"})

        assert out["amount"].to_list() == list(expected)
        assert out["amount"].dtype == pl.Float64

    def test_fill_down_never_crosses_into_another_group(self) -> None:
        """Account B's first row is blank and stays blank, though the row above it (account A) has a value."""
        frame = pl.DataFrame(
            {"account": ["A", "B", "A", "B", "A"], "amount": pl.Series([1.0, None, None, 4.0, None], dtype=pl.Float64)}
        )

        out = _run(
            frame,
            {"output": "amount", "expression": "coalesce([amount], [Row-1:amount])", "group_by": ["account"]},
        )

        assert out["amount"].to_list() == [1.0, None, 1.0, 4.0, 1.0]

    def test_a_running_balance_within_each_account(self) -> None:
        frame = pl.DataFrame({"account": ["A", "B", "A", "A", "B"], "amount": [100.0, 50.0, -30.0, 10.0, 25.0]})

        out = _run(
            frame,
            {
                "output": "balance",
                "type": "Float64",
                "expression": "coalesce([Row-1:balance], 0) + [amount]",
                "group_by": ["account"],
            },
        )

        assert out.columns == ["account", "amount", "balance"]
        assert out["balance"].dtype == pl.Float64
        assert out["balance"].to_list() == [100.0, 50.0, 70.0, 80.0, 75.0]

    def test_a_plain_reference_to_the_output_reads_the_current_rows_original_value(self) -> None:
        """``[amount]`` is this row's input value; ``[Row-1:amount]`` is the value written one row up.

        Reading the input one row up would give 10, 30, 25, 45; reading the written value for the
        current row is impossible. Only the rule as stated gives 10, 30, 35, 75.
        """
        frame = pl.DataFrame({"amount": [10.0, 20.0, 5.0, 40.0]})

        out = _run(frame, {"output": "amount", "expression": "[amount] + coalesce([Row-1:amount], 0)"})

        assert out["amount"].to_list() == [10.0, 30.0, 35.0, 75.0]


# ---------------------------------------------------------------------------
# The output's type
# ---------------------------------------------------------------------------


class TestOutputType:
    """``type`` is required only where it cannot be worked out, and the result must convert to it exactly."""

    @pytest.mark.parametrize("height", [3, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_new_self_referencing_column_without_a_type_is_refused(self, height: int) -> None:
        frame = pl.DataFrame({"amount": [1.0, 2.0, 3.0]}).head(height)

        with pytest.raises(ConfigError, match="type"):
            _run(frame, {"output": "balance", "expression": "coalesce([Row-1:balance], 0) + [amount]"})

    def test_a_new_column_that_does_not_read_itself_needs_no_type(self) -> None:
        """Mirror: without self-reference the type is the expression's own."""
        out = _run(pl.DataFrame({"amount": [1.0, 2.0]}), {"output": "previous", "expression": "[Row-1:amount]"})

        assert out["previous"].dtype == pl.Float64

    @pytest.mark.parametrize(
        ("type_settings", "quoted"),
        [
            pytest.param({"type": "Money"}, "Money", id="a_type_the_library_does_not_have"),
            pytest.param({"type": "Decimal"}, "places", id="decimal_without_its_places"),
        ],
    )
    def test_a_type_the_library_cannot_write_is_refused(self, type_settings: Mapping[str, Any], quoted: str) -> None:
        config = {"output": "balance", "expression": "coalesce([Row-1:balance], 0) + [amount]", **type_settings}

        with pytest.raises(ConfigError, match=quoted):
            _run(pl.DataFrame({"amount": [1.0, 2.0]}), config)

    @pytest.mark.parametrize(
        ("amounts", "expected"),
        [
            pytest.param([1.0, 2.0, 4.0], [1, 3, 7], id="whole_results_convert_to_a_whole_number_type"),
            pytest.param([1.5, 2.0, 4.0], DataError, id="a_fraction_does_not_fit_and_stops_the_step"),
        ],
    )
    def test_the_result_converts_to_the_stated_type_without_rounding(
        self, amounts: Sequence[float], expected: Sequence[int] | type[DataError]
    ) -> None:
        frame = pl.DataFrame({"amount": amounts})
        config = {"output": "balance", "type": "Int64", "expression": "coalesce([Row-1:balance], 0) + [amount]"}

        if expected is DataError:
            with pytest.raises(DataError):
                _run(frame, config)
            return
        out = _run(frame, config)

        assert out["balance"].dtype == pl.Int64
        assert out["balance"].to_list() == list(expected)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class TestRefusals:
    """Settings mistakes are refused naming what was found, read off the schema before any row."""

    @pytest.mark.parametrize(
        ("config", "quoted"),
        [
            pytest.param(
                {"output": "amount", "expression": "coalesce([amount], [Row+1:amount])"},
                "amount",
                id="a_later_row_of_the_existing_output_column",
            ),
            pytest.param(
                {"output": "balance", "type": "Float64", "expression": "[amount] + coalesce([Row+1:balance], 0)"},
                "balance",
                id="a_later_row_of_a_new_output_column",
            ),
            pytest.param({"output": "result", "expression": "[Row-0:amount]"}, "Row-0", id="a_reach_of_zero_rows"),
            pytest.param(
                {"output": "result", "expression": "[Row-1001:amount]"}, "1001", id="a_reach_past_a_thousand_rows"
            ),
            pytest.param(
                {"output": "result", "expression": "[Row-1:amonut]"},
                "amonut",
                id="a_row_reference_to_a_column_that_is_not_there",
            ),
            pytest.param(
                {"output": "result", "expression": "[Row-1:amount]", "group_by": ["acount"]},
                "acount",
                id="a_group_column_that_is_not_there",
            ),
            pytest.param(
                {"output": "result", "expression": "[amount] * [Row-1:rate]"},
                "rate",
                id="an_exact_value_meeting_a_float",
            ),
        ],
    )
    @pytest.mark.parametrize("height", [2, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_mistake_in_the_settings_is_refused_by_name(
        self, config: Mapping[str, Any], quoted: str, height: int
    ) -> None:
        frame = pl.DataFrame(
            {
                "amount": pl.Series([Decimal("1.00"), Decimal("2.00")], dtype=pl.Decimal(38, 2)),
                "rate": [1.0, 2.0],
            }
        ).head(height)

        with pytest.raises(ConfigError) as caught:
            _run(frame, config)

        assert quoted in str(caught.value), f"the refusal should quote {quoted!r}: {caught.value}"

    @pytest.mark.parametrize(
        ("config", "quantities", "expected"),
        [
            pytest.param(
                {"output": "per_unit", "expression": "[amount] / [Row-1:quantity]"},
                [2.0, 4.0, 0.0],
                [None, 10.0, 7.5],
                id="a_zero_no_row_divides_by_is_harmless",
            ),
            pytest.param(
                {"output": "per_unit", "expression": "[amount] / [Row-1:quantity]"},
                [2.0, 0.0, 4.0],
                DataError,
                id="a_zero_divisor_read_from_the_row_above_stops_the_step",
            ),
            pytest.param(
                {"output": "share", "type": "Float64", "expression": "coalesce([Row-1:share], [amount]) / [quantity]"},
                [2.0, 0.0, 4.0],
                DataError,
                id="a_zero_divisor_in_a_self_referencing_formula_stops_the_step",
            ),
        ],
    )
    def test_division_by_zero_on_a_row_stops_the_step(
        self,
        config: Mapping[str, Any],
        quantities: Sequence[float],
        expected: Sequence[float | None] | type[DataError],
    ) -> None:
        frame = pl.DataFrame({"amount": [10.0, 20.0, 30.0], "quantity": quantities})

        if expected is DataError:
            with pytest.raises(DataError):
                _run(frame, config)
            return
        out = _run(frame, config)

        assert out[config["output"]].to_list() == list(expected)


# ---------------------------------------------------------------------------
# Exact decimals, empty input, size
# ---------------------------------------------------------------------------


class TestExactDecimals:
    """Arithmetic on ``Decimal`` columns stays exact across rows."""

    def _amounts(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "amount": pl.Series(
                    [Decimal("0.10"), Decimal("0.20"), Decimal("0.30"), Decimal("0.40")], dtype=pl.Decimal(38, 2)
                )
            }
        )

    def test_a_running_balance_of_exact_amounts_is_exact(self) -> None:
        """In floats this would give 0.30000000000000004 and 0.6000000000000001."""
        out = _run(
            self._amounts(),
            {
                "output": "balance",
                "type": "Decimal",
                "places": 2,
                "expression": "coalesce([Row-1:balance], 0) + [amount]",
            },
        )

        assert out["balance"].dtype == pl.Decimal(38, 2)
        assert out["balance"].to_list() == [Decimal("0.10"), Decimal("0.30"), Decimal("0.60"), Decimal("1.00")]

    def test_a_change_between_exact_rows_is_exact(self) -> None:
        out = _run(self._amounts(), {"output": "change", "expression": "[amount] - [Row-1:amount]"})

        assert out["change"].dtype == pl.Decimal(38, 2)
        assert out["change"].to_list() == [None, Decimal("0.10"), Decimal("0.10"), Decimal("0.10")]


class TestEmptyInputAndSize:
    """No rows give no rows with the output column; a long self-referencing run gives the right answer."""

    @pytest.mark.parametrize(
        ("config", "dtype"),
        [
            pytest.param({"output": "change", "expression": "[amount] - [Row-1:amount]"}, pl.Float64, id="plain"),
            pytest.param(
                {"output": "balance", "type": "Int64", "expression": "coalesce([Row-1:balance], 0) + [amount]"},
                pl.Int64,
                id="self_referencing_with_a_type",
            ),
        ],
    )
    def test_empty_input_gives_empty_output_with_the_output_column(
        self, config: Mapping[str, Any], dtype: pl.DataType
    ) -> None:
        frame = pl.DataFrame({"account": pl.Series([], dtype=pl.String), "amount": pl.Series([], dtype=pl.Float64)})

        out = _run(frame, {**config, "group_by": ["account"]})

        assert out.height == 0
        assert out.columns == ["account", "amount", config["output"]]
        assert out[config["output"]].dtype == dtype

    def test_fifty_thousand_rows_of_fill_down(self) -> None:
        """A value on every hundredth row, blanks between; every blank takes the value above it."""
        size = 50_000
        frame = pl.DataFrame({"amount": pl.Series([float(i) if i % 100 == 0 else None for i in range(size)])})
        expected = [float(i - i % 100) for i in range(size)]

        started = time.perf_counter()
        out = _run(frame, {"output": "amount", "expression": "coalesce([amount], [Row-1:amount])"})
        elapsed = time.perf_counter() - started
        print(f"\nmulti_row_formula: {size:,} rows of self-reference fill-down in {elapsed:.2f}s")

        assert out.height == size
        assert out["amount"].to_list() == expected


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class TestCatalog:
    def test_the_tool_is_registered_as_a_transform_with_one_rows_port_each_way_and_conforms(self) -> None:
        tool = get_tool(TOOL_NAME)

        check_tool(tool)

        assert tool.name == TOOL_NAME
        assert tool.category == "transform"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",)

    def test_the_tool_leaves_its_input_untouched(self) -> None:
        frame = pl.DataFrame({"amount": pl.Series([5.0, None, 8.0], dtype=pl.Float64)})
        before = frame.clone()

        _run(frame, {"output": "amount", "expression": "coalesce([amount], [Row-1:amount])"})

        assert_frame_equal(frame, before)


# ---------------------------------------------------------------------------
# After the adversarial review: self-reference on both sides
# ---------------------------------------------------------------------------

BALANCE = "coalesce([Row-1:balance], 0) + [amount]"


def _balance(frame: pl.DataFrame, group_by: list[str] | None = None, **extra: Any) -> pl.DataFrame:
    """A running balance by self-reference, typed Float64 unless ``extra`` says otherwise."""
    config: dict[str, Any] = {"output": "balance", "type": "Float64", "expression": BALANCE, **extra}
    if group_by is not None:
        config["group_by"] = group_by
    return _run(frame, config)


class TestSelfReferenceBeyondTheRowAbove:
    """``[Row-N:<output>]`` for N above 1, and two self-references in one expression."""

    FIBONACCI = "coalesce([Row-1:n], 1) + coalesce([Row-2:n], 0)"

    def test_two_self_references_at_once_build_each_value_from_the_two_written_above(self) -> None:
        out = _run(
            pl.DataFrame({"key": list(range(7))}), {"output": "n", "type": "Int64", "expression": self.FIBONACCI}
        )

        assert out["n"].dtype == pl.Int64
        assert out["n"].to_list() == [1, 1, 2, 3, 5, 8, 13]

    def test_two_self_references_at_once_stay_inside_each_group(self) -> None:
        """A in rows 0, 2, 4, 6 gives 1, 1, 2, 3; B in rows 1, 3, 5 gives 1, 1, 2."""
        frame = pl.DataFrame({"account": ["A", "B", "A", "B", "A", "B", "A"]})

        out = _run(frame, {"output": "n", "type": "Int64", "expression": self.FIBONACCI, "group_by": ["account"]})

        assert out["n"].to_list() == [1, 1, 1, 1, 2, 2, 3]

    def test_a_self_reference_three_hundred_rows_up(self) -> None:
        """Each row adds its amount to the total written 300 rows above it: 300 separate running totals."""
        size, reach = 1_200, 300
        expected: list[int] = []
        for i in range(size):
            expected.append(i + (expected[i - reach] if i >= reach else 0))

        out = _run(
            pl.DataFrame({"amount": list(range(size))}),
            {"output": "total", "type": "Int64", "expression": f"coalesce([Row-{reach}:total], 0) + [amount]"},
        )

        assert out["total"].to_list() == expected


class TestTwoGroupColumns:
    """``group_by`` with two columns: a group is the pair, and a null in one of them is a value of the pair."""

    def _frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "region": ["N", "S", "N", "S", "N", "N"],
                "account": ["A", "A", "A", "A", None, None],
                "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )

    def test_a_running_balance_keeps_pairs_apart_and_puts_the_two_null_account_rows_together(self) -> None:
        out = _balance(self._frame(), ["region", "account"])

        assert out["balance"].to_list() == [1.0, 2.0, 4.0, 6.0, 5.0, 11.0]

    def test_a_plain_row_reference_groups_the_same_way(self) -> None:
        out = _run(
            self._frame(), {"output": "previous", "expression": "[Row-1:amount]", "group_by": ["region", "account"]}
        )

        assert out["previous"].to_list() == [None, None, 1.0, 2.0, None, 5.0]


class TestLongRuns:
    """Runs of thousands of rows that depend on each other give the right answer, whatever the run's length."""

    def test_one_value_then_three_thousand_blanks_fills_every_row(self) -> None:
        size = 3_001
        frame = pl.DataFrame(
            {"account": ["A"] * size, "amount": pl.Series([7.0] + [None] * (size - 1), dtype=pl.Float64)}
        )

        out = _run(
            frame, {"output": "amount", "expression": "coalesce([amount], [Row-1:amount])", "group_by": ["account"]}
        )

        assert out["amount"].to_list() == [7.0] * size

    def test_a_running_balance_over_three_thousand_rows_ends_at_the_total(self) -> None:
        amounts = [float(i % 17) for i in range(3_000)]
        expected: list[float] = []
        for amount in amounts:
            expected.append((expected[-1] if expected else 0.0) + amount)

        out = _balance(pl.DataFrame({"account": ["A"] * len(amounts), "amount": amounts}), ["account"])

        assert out["balance"].to_list() == expected
        assert out["balance"].to_list()[-1] == sum(amounts)

    def test_a_short_group_arriving_after_a_long_one_is_worked_out_on_its_own(self) -> None:
        frame = pl.DataFrame({"account": ["A"] * 2_500 + ["B"] * 3, "amount": [1.0] * 2_500 + [10.0, 20.0, 30.0]})

        balances = _balance(frame, ["account"])["balance"].to_list()

        assert balances[2_499] == 2_500.0
        assert balances[-3:] == [10.0, 30.0, 60.0]


class TestOtherColumnsInsideASelfReference:
    """``[Row±N:<other column>]`` in a self-referencing formula reads the input, within the group."""

    @pytest.mark.parametrize(
        ("reference", "expected"),
        [
            pytest.param("[Row+1:amount]", [2.0, 20.0, 5.0, 20.0, 5.0], id="the_next_rows_amount"),
            pytest.param("[Row-1:amount]", [0.0, 0.0, 1.0, 10.0, 3.0], id="the_previous_rows_amount"),
        ],
    )
    def test_another_columns_row_reference_beside_a_self_reference(
        self, reference: str, expected: Sequence[float]
    ) -> None:
        frame = pl.DataFrame({"account": ["A", "B", "A", "B", "A"], "amount": [1.0, 10.0, 2.0, 20.0, 3.0]})

        out = _run(
            frame,
            {
                "output": "t",
                "type": "Float64",
                "expression": f"coalesce([Row-1:t], 0) + coalesce({reference}, 0)",
                "group_by": ["account"],
            },
        )

        assert out["t"].to_list() == list(expected)


class TestStopsInsideASelfReference:
    """A row with no answer stops a self-referencing step, naming the rows ``formula`` would name."""

    def _exact(self, values: Sequence[str]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "key": ["a", "b", "c"][: len(values)],
                "amount": pl.Series([Decimal(v) for v in values], dtype=pl.Decimal(38, 2)),
            }
        )

    def test_an_exact_balance_past_38_digits_stops_the_step_naming_the_second_row(self) -> None:
        big = "900000000000000000000000000000000000.00"  # 36 digits before the point: the most 2 places allow
        frame = self._exact([big, big, "1.00"])

        with pytest.raises(DataError) as caught:
            _balance(frame, type="Decimal", places=2)

        keys = [row["key"] for row in caught.value.examples]
        assert keys[0] == "b", f"the second row is the first without an answer: {caught.value}"
        assert "a" not in keys, f"the first row has an answer: {caught.value}"
        for row in caught.value.examples:
            assert set(row) == set(frame.columns), f"only the input's own columns: {row}"

    def test_an_exact_balance_of_small_amounts_runs(self) -> None:
        out = _balance(self._exact(["1.00", "2.00", "3.00"]), type="Decimal", places=2)

        assert out["balance"].to_list() == [Decimal("1.00"), Decimal("3.00"), Decimal("6.00")]

    def test_a_divisor_that_is_zero_only_before_the_rows_above_are_settled_does_not_stop_the_step(self) -> None:
        """Settled: 10 / (0 + 1) = 10, then 20 / (10 + 0) = 2, then 30 / (2 + 0) = 15."""
        frame = pl.DataFrame({"amount": [10.0, 20.0, 30.0], "k": [1.0, 0.0, 0.0]})

        out = _run(frame, {"output": "x", "type": "Float64", "expression": "[amount] / (coalesce([Row-1:x], 0) + [k])"})

        assert out["x"].to_list() == [10.0, 2.0, 15.0]

    def test_a_division_by_zero_names_every_offending_row_in_arrival_order_with_the_inputs_columns_only(
        self,
    ) -> None:
        frame = pl.DataFrame(
            {
                "key": ["a", "b", "c", "d", "e"],
                "account": ["A", "B", "A", "B", "A"],
                "amount": [10.0, 10.0, 10.0, 10.0, 10.0],
                "quantity": [2.0, 0.0, 5.0, 4.0, 0.0],
            }
        )

        with pytest.raises(DataError) as caught:
            _run(
                frame,
                {
                    "output": "share",
                    "type": "Float64",
                    "expression": "coalesce([Row-1:share], 0) + [amount] / [quantity]",
                    "group_by": ["account"],
                },
            )

        error = caught.value
        assert [row["key"] for row in error.examples] == ["b", "e"], f"both zero rows, in order: {error}"
        assert error.offending_rows == 2
        for row in error.examples:
            assert set(row) == set(frame.columns), f"only the input's own columns: {row}"


class TestTypeWithoutSelfReference:
    """``type`` converts the result on the plain path too, by ``cast_columns``' rules."""

    def test_whole_values_convert_to_a_stated_whole_number_type(self) -> None:
        out = _run(
            pl.DataFrame({"amount": [1.0, 2.0]}), {"output": "prev", "type": "Int64", "expression": "[Row-1:amount]"}
        )

        assert out["prev"].dtype == pl.Int64
        assert out["prev"].to_list() == [None, 1]

    def test_a_fraction_into_a_whole_number_type_stops_the_step(self) -> None:
        with pytest.raises(DataError):
            _run(
                pl.DataFrame({"amount": [1.0, 2.5, 3.0]}),
                {"output": "prev", "type": "Int64", "expression": "[Row-1:amount]"},
            )


class TestNotANumberFinishes:
    """A NaN in a self-referencing formula must finish, never loop, on every engine version."""

    def test_a_running_balance_through_a_nan_finishes(self) -> None:
        frame = pl.DataFrame({"amount": [1.0, float("nan"), 2.0]})
        outcome: dict[str, Any] = {}

        def work() -> None:
            try:
                outcome["out"] = _balance(frame)
            except BaseException as raised:  # noqa: BLE001 - handed back to the test thread
                outcome["error"] = raised

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        worker.join(timeout=60)

        assert not worker.is_alive(), "the step did not finish within 60 seconds"
        if "error" in outcome:
            raise outcome["error"]
        balances = outcome["out"]["balance"].to_list()
        assert balances[0] == 1.0
        assert math.isnan(balances[1]) and math.isnan(balances[2]), balances


class TestMinorRulings:
    """Cheap pins from the review: the runner's column check, settings refusals, types, and neighbouring tools."""

    def test_a_misspelt_plain_column_beside_a_row_reference_is_refused_by_the_runner(self) -> None:
        spec = load_pipeline(
            {
                "name": "a running balance",
                "steps": [
                    {"id": "entries", "tool": "read_csv", "config": {"path": "entries.csv"}},
                    {
                        "id": "balanced",
                        "tool": TOOL_NAME,
                        "config": {
                            "output": "balance",
                            "type": "Float64",
                            "expression": "coalesce([Row-1:amount], 0) + [amonut]",
                        },
                        "inputs": {"rows": "entries.rows"},
                    },
                    {
                        "id": "checked",
                        "tool": "assert_rows",
                        "config": {"predicate": "[balance] >= 0", "allow_empty": True},
                        "inputs": {"rows": "balanced.rows"},
                    },
                ],
            }
        )
        sources = {PortRef("entries", "rows"): pl.DataFrame({"amount": [5.0, 7.0]})}

        with pytest.raises(ColumnCheckError) as raised:
            run_pipeline(spec, sources=sources, retain="all")

        assert raised.value.step_id == "balanced"
        assert "amonut" in str(raised.value)

    @pytest.mark.parametrize(
        ("extra", "quoted"),
        [
            pytest.param({"group_by": ["account", "account"]}, "account", id="a_group_column_named_twice"),
            pytest.param({"places": 2}, "places", id="places_without_a_type"),
        ],
    )
    def test_a_repeated_group_column_and_places_without_a_type_are_refused(
        self, extra: Mapping[str, Any], quoted: str
    ) -> None:
        frame = pl.DataFrame({"account": ["A", "B"], "amount": [1.0, 2.0]})

        with pytest.raises(ConfigError, match=quoted):
            _run(frame, {"output": "previous", "expression": "[Row-1:amount]", **extra})

    @pytest.mark.parametrize(
        ("values", "extra", "expected", "dtype"),
        [
            pytest.param([2, 4], {}, [3, 9], pl.Int64, id="whole_results_keep_the_existing_whole_number_type"),
            pytest.param([1, 2], {}, DataError, None, id="a_fraction_into_the_existing_whole_number_type_stops"),
            pytest.param(
                [1, 2], {"type": "Float64"}, [1.5, 4.5], pl.Float64, id="a_stated_type_overrides_the_existing_one"
            ),
        ],
    )
    def test_an_existing_whole_number_column_written_by_self_reference(
        self,
        values: Sequence[int],
        extra: Mapping[str, Any],
        expected: Sequence[float] | type[DataError],
        dtype: pl.DataType | None,
    ) -> None:
        frame = pl.DataFrame({"x": pl.Series(values, dtype=pl.Int64)})
        config = {"output": "x", "expression": "coalesce([Row-1:x], 0) + [x] * 1.5", **extra}

        if expected is DataError:
            with pytest.raises(DataError):
                _run(frame, config)
            return
        out = _run(frame, config)

        assert out["x"].dtype == dtype
        assert out["x"].to_list() == list(expected)

    def test_formula_still_reads_a_column_literally_named_like_a_row_reference(self) -> None:
        frame = pl.DataFrame({"Row-1:x": [1.0, 2.0]})

        out = run_tool(
            get_tool("formula"), Ports({"rows": frame}), {"columns": [{"name": "y", "value": "[Row-1:x] * 2"}]}
        )["rows"]

        assert out["y"].to_list() == [2.0, 4.0]

    @pytest.mark.parametrize(
        ("tool", "config"),
        [
            pytest.param("formula", {"columns": [{"name": "y", "value": "[Row-1:x]"}]}, id="formula"),
            pytest.param("filter", {"predicate": "[Row-1:x] > 0"}, id="filter"),
            pytest.param(
                "case_when", {"cases": [{"when": "[Row-1:x] > 0", "then": "'up'"}], "output": "y"}, id="case_when"
            ),
        ],
    )
    def test_other_expression_tools_refuse_a_row_reference(self, tool: str, config: Mapping[str, Any]) -> None:
        with pytest.raises(ConfigError, match="Row-1:x"):
            run_tool(get_tool(tool), Ports({"rows": pl.DataFrame({"x": [1.0, 2.0]})}), dict(config))
