"""Tests for the running_total tool: a running sum per column, in arrival order, per group if asked.

The tool takes one input port, ``rows``, and gives one, ``rows``: every input row and column, in the
same order, plus one new column per entry in ``columns``, appended in the order listed. It mirrors
SQL's ``SUM(col) OVER (PARTITION BY group ORDER BY <arrival> ROWS UNBOUNDED PRECEDING)``.

What these tests pin:

- The worked example from the tool's requirement, with the two accounts' rows interleaved.
- No ``group_by`` gives one total over every row; one group column next to two.
- A null group value is a group of its own, as in SQL's PARTITION BY.
- Nulls as in SQL: a null adds nothing, and a total is null only until its group's first value.
  A group of nothing but nulls stays null.
- Types: whole numbers give Int64, floats give Float64, an exact Decimal gives an exact Decimal at
  its places with 38 digits.
- A running value past Int64's range, or past 38 digits, stops the step naming the group and the
  column. Every running value counts, not just the last. The values just inside the range are kept.
- Settings refusals, read off the schema, so they hold on a frame with no rows too.
- Row order, every input column and its type are kept; the input frame is not changed.
- No rows in gives no rows out, with the new columns typed.

The tool is reached only by name, through ``get_tool``, so these tests do not presume its module.
"""

from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from typing import Any, Final

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl import get_tool
from dantron_etl.authoring import CATEGORIES, ConfigError, DataError, Ports, check_tool, run_tool

_INT64_MAX: Final[int] = 2**63 - 1
_INT64_MIN: Final[int] = -(2**63)

#: The largest exact figure at 2 places: 36 nines before the point, 38 digits in all.
_LARGEST_AT_2: Final[str] = "9" * 36 + ".99"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(frame: pl.DataFrame, config: Mapping[str, Any]) -> pl.DataFrame:
    """Run running_total on ``frame`` and return its one output port."""
    result = run_tool(get_tool("running_total"), Ports({"rows": frame}), config)
    assert result.names == ("rows",), "one input, one output, both named rows"
    return result["rows"]


def _entry(column: str, output: str) -> dict[str, str]:
    """One entry of ``columns``."""
    return {"column": column, "output": output}


def _exact(values: list[str | None], places: int, *, precision: int = 38) -> pl.Series:
    """Exact decimals written as text, so nothing passes through a float on the way in."""
    return pl.Series([None if v is None else Decimal(v) for v in values], dtype=pl.Decimal(precision, places))


def _written(series: pl.Series) -> list[str | None]:
    """Each value as the digits it holds, so a test pins the places as well as the value."""
    return series.cast(pl.String).to_list()


def _ledger() -> pl.DataFrame:
    """The worked example's input: two accounts, their rows interleaved, amounts exact at 2 places."""
    return pl.DataFrame(
        {
            "account": ["A", "A", "B", "A", "B"],
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 5), date(2024, 1, 3)],
            "amount": _exact(["100.00", "-30.00", "20.00", "50.00", "5.00"], 2, precision=12),
        }
    )


# ---------------------------------------------------------------------------
# The worked example and grouping
# ---------------------------------------------------------------------------


class TestTheWorkedExample:
    """The example in the requirement, exactly."""

    def test_each_account_keeps_its_own_balance_when_its_rows_are_interleaved(self) -> None:
        """A's rows are 1, 2 and 4; B's are 3 and 5. Each total runs over its own rows only."""
        frame = _ledger()

        produced = _run(frame, {"group_by": ["account"], "columns": [_entry("amount", "balance")]})

        assert _written(produced["balance"]) == ["100.00", "70.00", "20.00", "120.00", "25.00"]
        assert produced.schema["balance"] == pl.Decimal(38, 2), "exact at the column's places, 38 digits"
        assert produced.columns == ["account", "date", "amount", "balance"]
        assert_frame_equal(produced.drop("balance"), frame)


class TestGrouping:
    """Where the total starts again."""

    def test_without_group_by_one_total_runs_over_every_row(self) -> None:
        """The same rows as the worked example, but nothing restarts the total."""
        frame = _ledger()

        produced = _run(frame, {"columns": [_entry("amount", "balance")]})

        assert _written(produced["balance"]) == ["100.00", "70.00", "90.00", "140.00", "145.00"]

    @pytest.mark.parametrize(
        ("group_by", "expected"),
        [
            pytest.param(["account"], [1, 3, 6, 4, 11], id="one_column_puts_both_regions_of_A_together"),
            pytest.param(["region", "account"], [1, 2, 4, 4, 7], id="two_columns_make_a_group_of_each_pair"),
        ],
    )
    def test_a_group_is_every_group_column_taken_together(self, group_by: list[str], expected: list[int]) -> None:
        """Account A appears in two regions. Grouped by account alone it is one group; with region, two."""
        frame = pl.DataFrame(
            {
                "region": ["north", "south", "north", "north", "south"],
                "account": ["A", "A", "A", "B", "A"],
                "amount": [1, 2, 3, 4, 5],
            }
        )

        produced = _run(frame, {"group_by": group_by, "columns": [_entry("amount", "total")]})

        assert produced["total"].to_list() == expected

    @pytest.mark.parametrize(
        ("groups", "amounts", "expected"),
        [
            pytest.param(
                {"account": ["A", None, "A", None, "B"]},
                [1, 10, 2, 20, 100],
                [1, 10, 3, 30, 100],
                id="the_null_rows_add_up_together_apart_from_every_named_group",
            ),
            pytest.param(
                {"region": [None, "north", None, None], "account": ["A", "A", "A", None]},
                [1, 10, 100, 1000],
                [1, 10, 101, 1000],
                id="a_null_in_one_of_two_columns_is_still_its_own_group",
            ),
        ],
    )
    def test_a_null_group_value_is_a_group_of_its_own(
        self,
        groups: dict[str, list[str | None]],
        amounts: list[int],
        expected: list[int],
    ) -> None:
        """As SQL's PARTITION BY: the null rows share one total. They are not merged into another group,
        and they do not each start again. With two columns, (null, A) and (null, null) are two groups."""
        frame = pl.DataFrame({**groups, "amount": amounts})

        produced = _run(frame, {"group_by": list(groups), "columns": [_entry("amount", "total")]})

        assert produced["total"].to_list() == expected


# ---------------------------------------------------------------------------
# Nulls
# ---------------------------------------------------------------------------


class TestNulls:
    """A null adds nothing. A total is null only until its group's first value."""

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            pytest.param(pl.Series([None, 5, None, 3], dtype=pl.Int64), [None, 5, 5, 8], id="whole_numbers"),
            pytest.param(
                pl.Series([None, 0.5, None, 0.25], dtype=pl.Float64),
                [None, 0.5, 0.5, 0.75],
                id="floats",
            ),
            pytest.param(
                _exact([None, "5.00", None, "3.00"], 2),
                [None, "5.00", "5.00", "8.00"],
                id="exact_decimals",
            ),
        ],
    )
    def test_a_null_adds_nothing_and_leads_to_null_only_before_the_first_value(
        self,
        values: pl.Series,
        expected: list[Any],
    ) -> None:
        """``[null, 5, null, 3]`` gives ``[null, 5, 5, 8]``: the first row has nothing yet, the third
        keeps 5. A null is not a zero (the first row would be 0) and does not poison the rest
        (every row after it would be null)."""
        frame = pl.DataFrame({"amount": values})

        produced = _run(frame, {"columns": [_entry("amount", "total")]})

        got = _written(produced["total"]) if values.dtype == pl.Decimal(38, 2) else produced["total"].to_list()
        assert got == expected

    def test_a_group_of_only_nulls_stays_null_beside_a_group_with_values(self) -> None:
        """B has values, A has none. A's totals are null, not zero; B's are untouched by A's nulls."""
        frame = pl.DataFrame({"account": ["A", "B", "A", "B"], "amount": pl.Series([None, 1, None, 2], dtype=pl.Int64)})

        produced = _run(frame, {"group_by": ["account"], "columns": [_entry("amount", "total")]})

        assert produced["total"].to_list() == [None, 1, None, 3]


# ---------------------------------------------------------------------------
# Several entries and result types
# ---------------------------------------------------------------------------


class TestSeveralEntries:
    """Each entry is its own running total, and the new columns go on in the order listed."""

    def test_two_entries_run_separately_and_are_appended_in_the_order_listed(self) -> None:
        """``fee`` is listed first, so its total comes first, even though ``amount`` is first in the frame.
        Each total has its own nulls: a null fee does not hold back the amount total."""
        frame = pl.DataFrame(
            {
                "account": ["A", "B", "A", "B"],
                "amount": pl.Series([10, 20, 30, 40], dtype=pl.Int64),
                "fee": pl.Series([None, 0.5, 1.5, None], dtype=pl.Float64),
            }
        )

        produced = _run(
            frame,
            {"group_by": ["account"], "columns": [_entry("fee", "fee_total"), _entry("amount", "amount_total")]},
        )

        assert produced.columns == ["account", "amount", "fee", "fee_total", "amount_total"]
        assert produced["fee_total"].to_list() == [None, 0.5, 1.5, 0.5]
        assert produced["amount_total"].to_list() == [10, 20, 40, 60]
        assert produced.schema["fee_total"] == pl.Float64
        assert produced.schema["amount_total"] == pl.Int64


class TestResultTypes:
    """Whole numbers give Int64, floats give Float64, exact decimals stay exact."""

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            pytest.param(pl.Series([3, -1, 7], dtype=pl.Int64), [3, 2, 9], id="int64"),
            pytest.param(
                pl.Series([2**31 - 1, 1], dtype=pl.Int32),
                [2**31 - 1, 2**31],
                id="int32_widens_to_int64_rather_than_wrapping",
            ),
            pytest.param(pl.Series([100, 100], dtype=pl.Int8), [100, 200], id="int8_widens_to_int64"),
        ],
    )
    def test_a_whole_number_column_gives_int64_totals(self, values: pl.Series, expected: list[int]) -> None:
        """A narrower whole-number column would wrap at its own limit; the total is Int64 instead."""
        produced = _run(pl.DataFrame({"amount": values}), {"columns": [_entry("amount", "total")]})

        assert produced.schema["total"] == pl.Int64
        assert produced["total"].to_list() == expected
        assert produced.schema["amount"] == values.dtype, "the input column keeps its own type"

    @pytest.mark.parametrize(
        "dtype",
        [pytest.param(pl.Float64, id="float64"), pytest.param(pl.Float32, id="float32_widens_to_float64")],
    )
    def test_a_float_column_gives_float64_totals(self, dtype: pl.DataType) -> None:
        """Values a float holds exactly, so the expected totals are exact too."""
        frame = pl.DataFrame({"amount": pl.Series([0.5, 0.25, -1.0], dtype=dtype)})

        produced = _run(frame, {"columns": [_entry("amount", "total")]})

        assert produced.schema["total"] == pl.Float64
        assert produced["total"].to_list() == [0.5, 0.75, -0.25]

    @pytest.mark.parametrize(
        ("places", "precision", "values", "expected"),
        [
            pytest.param(
                2,
                10,
                ["0.10", "0.20", "0.30"],
                ["0.10", "0.30", "0.60"],
                id="two_places_exact_where_a_float_would_not_be",
            ),
            pytest.param(
                4,
                38,
                ["0.0725", "-1.0000", "0.0001"],
                ["0.0725", "-0.9275", "-0.9274"],
                id="four_places",
            ),
            pytest.param(0, 20, ["7", "-2", "10"], ["7", "5", "15"], id="no_places"),
        ],
    )
    def test_an_exact_column_gives_exact_totals_at_its_places_with_38_digits(
        self,
        places: int,
        precision: int,
        values: list[str],
        expected: list[str],
    ) -> None:
        """0.10 + 0.20 is 0.30 here, which a float total would not give. The places are kept, and the
        total has 38 digits whatever the input column had."""
        frame = pl.DataFrame({"amount": _exact(values, places, precision=precision)})

        produced = _run(frame, {"columns": [_entry("amount", "total")]})

        assert produced.schema["total"] == pl.Decimal(38, places)
        assert _written(produced["total"]) == expected


# ---------------------------------------------------------------------------
# Range
# ---------------------------------------------------------------------------


class TestATotalPastItsRangeStopsTheStep:
    """Every running value must fit, not only the last. One that does not stops the step by name."""

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param([_INT64_MAX, 1], id="one_past_the_ceiling"),
            pytest.param([_INT64_MIN, -1], id="one_past_the_floor"),
            pytest.param([2**62, 2**62], id="two_values_that_fit_alone"),
            pytest.param([_INT64_MAX, 1, -1], id="past_the_ceiling_then_back_inside"),
        ],
    )
    def test_an_int64_total_past_its_range_stops_naming_the_group_and_the_column(self, values: list[int]) -> None:
        """The engine wraps a total past Int64's range round to the other end and says nothing. The last
        case ends inside the range, but its middle row's total does not fit, so it stops too. The other
        group fits and does not stop the step on its own; the refusal names the one that broke."""
        frame = pl.DataFrame(
            {
                "account": ["acct-7"] * len(values) + ["acct-2"],
                "amount": pl.Series([*values, 1], dtype=pl.Int64),
            }
        )

        with pytest.raises(DataError) as caught:
            _run(frame, {"group_by": ["account"], "columns": [_entry("amount", "total")]})

        message = str(caught.value)
        assert "acct-7" in message, f"the refusal should name the group: {message}"
        assert "amount" in message, f"the refusal should name the column: {message}"

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            pytest.param([_INT64_MAX - 1, 1], [_INT64_MAX - 1, _INT64_MAX], id="ends_on_the_ceiling"),
            pytest.param([_INT64_MIN + 1, -1], [_INT64_MIN + 1, _INT64_MIN], id="ends_on_the_floor"),
            pytest.param([_INT64_MAX, -1, 1], [_INT64_MAX, _INT64_MAX - 1, _INT64_MAX], id="touches_the_ceiling_twice"),
            pytest.param([_INT64_MAX, _INT64_MIN], [_INT64_MAX, -1], id="ceiling_then_floor"),
        ],
    )
    def test_an_int64_total_just_inside_its_range_is_kept(self, values: list[int], expected: list[int]) -> None:
        """The mirror: a check that refused one value too early would break a total that lands exactly
        on the limit."""
        frame = pl.DataFrame({"amount": pl.Series(values, dtype=pl.Int64)})

        produced = _run(frame, {"columns": [_entry("amount", "total")]})

        assert produced["total"].to_list() == expected
        assert produced.schema["total"] == pl.Int64

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param([_LARGEST_AT_2, "0.01"], id="one_cent_over"),
            pytest.param(["-" + _LARGEST_AT_2, "-0.01"], id="one_cent_under"),
            pytest.param(["9" * 36 + ".00", "9" * 36 + ".00", "-" + "9" * 36 + ".00"], id="over_then_back_inside"),
        ],
    )
    def test_an_exact_total_past_38_digits_stops_naming_the_group_and_the_column(self, values: list[str]) -> None:
        """Unlike a group's final sum, a running total shows every step, so a middle step past 38 digits
        stops the step even when the last one fits."""
        frame = pl.DataFrame(
            {
                "account": ["acct-7"] * len(values) + ["acct-2"],
                "amount": _exact([*values, "1.00"], 2),
            }
        )

        with pytest.raises(DataError) as caught:
            _run(frame, {"group_by": ["account"], "columns": [_entry("amount", "total")]})

        message = str(caught.value)
        assert "acct-7" in message, f"the refusal should name the group: {message}"
        assert "amount" in message, f"the refusal should name the column: {message}"

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            pytest.param(["9" * 36 + ".98", "0.01"], ["9" * 36 + ".98", _LARGEST_AT_2], id="ends_on_the_largest"),
            pytest.param(
                ["-" + "9" * 36 + ".98", "-0.01"],
                ["-" + "9" * 36 + ".98", "-" + _LARGEST_AT_2],
                id="ends_on_the_smallest",
            ),
        ],
    )
    def test_an_exact_total_just_inside_38_digits_is_kept(self, values: list[str], expected: list[str]) -> None:
        """The mirror of the refusal above, one cent inside the limit."""
        frame = pl.DataFrame({"amount": _exact(values, 2)})

        produced = _run(frame, {"columns": [_entry("amount", "total")]})

        assert _written(produced["total"]) == expected
        assert produced.schema["total"] == pl.Decimal(38, 2)


# ---------------------------------------------------------------------------
# Rows and columns kept
# ---------------------------------------------------------------------------


def _mixed() -> pl.DataFrame:
    """A frame in no useful order, with several types and a null in each nullable column."""
    return pl.DataFrame(
        {
            "code": ["d", "a", "c", "b"],
            "account": ["north", None, "north", "east"],
            "amount": pl.Series([10, -5, None, 2], dtype=pl.Int64),
            "rate": pl.Series([0.5, None, 1.5, 2.0], dtype=pl.Float64),
            "price": _exact(["1.25", "2.50", None, "-0.75"], 2, precision=10),
            "active": [True, False, None, True],
            "opened": [date(2024, 3, 1), date(2024, 1, 1), None, date(2024, 2, 1)],
        }
    )


class TestRowsAndColumnsAreKept:
    """Nothing moves, nothing is dropped, and the input frame is not changed."""

    def test_row_order_and_every_input_column_and_its_type_are_kept(self) -> None:
        """The rows are in no sorted order; the tool must not sort them, by group or otherwise."""
        frame = _mixed()
        before = frame.clone()

        produced = _run(frame, {"group_by": ["account"], "columns": [_entry("amount", "running")]})

        assert produced.columns == [*frame.columns, "running"], "the new column goes on the end"
        assert_frame_equal(produced.drop("running"), frame)
        assert produced["code"].to_list() == ["d", "a", "c", "b"], "arrival order, not group order"
        assert produced["running"].to_list() == [10, -5, 10, 2]
        assert_frame_equal(frame, before)

    def test_no_rows_in_gives_no_rows_out_with_the_new_columns_typed(self) -> None:
        """Nothing to infer a type from, so each new column's type comes from its source column."""
        empty = _mixed().head(0)

        produced = _run(
            empty,
            {
                "group_by": ["account"],
                "columns": [_entry("amount", "t_amount"), _entry("rate", "t_rate"), _entry("price", "t_price")],
            },
        )

        assert produced.height == 0
        assert produced.schema == {
            **empty.schema,
            "t_amount": pl.Int64,
            "t_rate": pl.Float64,
            "t_price": pl.Decimal(38, 2),
        }


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def _refusal_cases() -> list[Any]:
    """(config, the text its refusal must quote) for each way the settings can be wrong."""
    return [
        pytest.param({"columns": [_entry("code", "total")]}, "code", id="a_text_column"),
        pytest.param({"columns": [_entry("active", "total")]}, "active", id="a_true_false_column"),
        pytest.param({"columns": [_entry("opened", "total")]}, "opened", id="a_date_column"),
        pytest.param({"columns": [_entry("amount", "rate")]}, "rate", id="an_output_name_the_input_already_has"),
        pytest.param(
            {"columns": [_entry("amount", "total"), _entry("rate", "total")]},
            "total",
            id="an_output_name_used_twice",
        ),
        pytest.param({"columns": []}, "columns", id="an_empty_columns_list"),
        pytest.param({}, "columns", id="no_columns_at_all"),
        pytest.param({"columns": [_entry("nowhere", "total")]}, "nowhere", id="a_column_the_input_does_not_have"),
        pytest.param(
            {"group_by": ["nowhere"], "columns": [_entry("amount", "total")]},
            "nowhere",
            id="a_group_column_the_input_does_not_have",
        ),
    ]


class TestRefusals:
    """Settings problems are refused before any row is read, naming the field or column to fix."""

    @pytest.mark.parametrize(("config", "named"), _refusal_cases())
    @pytest.mark.parametrize("height", [4, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_settings_the_input_cannot_carry_are_refused_by_name(
        self,
        config: Mapping[str, Any],
        named: str,
        height: int,
    ) -> None:
        """Each case runs on a frame with rows and on one without, which shows the refusal comes from
        the column names and types, not the values."""
        frame = _mixed().head(height)

        with pytest.raises(ConfigError) as caught:
            _run(frame, config)

        message = str(caught.value)
        assert named in message, f"the refusal should name {named!r}: {message}"

    @pytest.mark.parametrize(
        ("column", "dtype"),
        [
            pytest.param("amount", pl.Int64, id="whole"),
            pytest.param("rate", pl.Float64, id="float"),
            pytest.param("price", pl.Decimal(38, 2), id="exact"),
        ],
    )
    def test_each_kind_of_number_is_accepted(self, column: str, dtype: pl.DataType) -> None:
        """The mirror of the refusals above: whole, float and exact columns all pass the settings check."""
        produced = _run(_mixed(), {"columns": [_entry(column, "total")]})

        assert produced.schema["total"] == dtype


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class TestRunningTotalInTheCatalog:
    """Registered by name, a transform, and its worked examples run."""

    def test_running_total_conforms_and_is_a_transform_with_one_rows_port(self) -> None:
        """Conformance runs every worked example in ``help()``, so a documented total is a checked one.
        Row order is part of the promise, so the examples are compared in order."""
        tool = get_tool("running_total")

        check_tool(tool)

        assert tool.name == "running_total"
        assert tool.category == "transform"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",)
        examples = tool.help().examples
        assert examples, "a tool nobody can copy from is a tool nobody uses"
        assert all(example.check_row_order for example in examples)
