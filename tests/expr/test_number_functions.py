"""floor, ceil, sign, mod, least, greatest and nullif: each gives what PostgreSQL gives.

mod takes the sign of its first argument, as SQL and Alteryx do, where the engine's own % takes the
sign of the second. mod by zero stops the step, as a division by zero does. least and greatest skip
nulls. floor, ceil and sign keep the number's own type.
"""

import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import polars as pl
import pytest

from dantron_etl.authoring import REGISTRY, DataError, Ports, run_tool
from dantron_etl.expr import ExpressionError, compile_expr


def _values(frame: pl.DataFrame, text: str) -> list[Any]:
    return frame.select(compile_expr(text, frame.schema).alias("out"))["out"].to_list()


def _dtype(frame: pl.DataFrame, text: str) -> pl.DataType:
    return frame.select(compile_expr(text, frame.schema).alias("out"))["out"].dtype


def _refusal(frame: pl.DataFrame, text: str) -> str:
    with pytest.raises(ExpressionError) as raised:
        compile_expr(text, frame.schema)
    return str(raised.value)


def _formula(frame: pl.DataFrame, value: str) -> list[Any]:
    config = {"columns": [{"name": "out", "value": value}]}
    return run_tool(REGISTRY.get("formula"), Ports({"rows": frame}), config)["rows"]["out"].to_list()


NUMBERS = pl.DataFrame(
    {
        "x": [-2.5, 2.5, -0.5, 0.5, -0.0, None, float("nan")],
        "n": [-3, 3, 0, 1, 0, None, 7],
        "small": pl.Series([-3, 3, 0, 1, 0, None, 7], dtype=pl.Int32),
        "f32": pl.Series([-2.5, 2.5, -0.5, 0.5, -0.0, None, 1.0], dtype=pl.Float32),
        "money": pl.Series(
            [Decimal("-1.25"), Decimal("1.25"), Decimal("-0.50"), Decimal("0.50"), Decimal("0"), None, Decimal("7")],
            dtype=pl.Decimal(10, 2),
        ),
        "code": ["a", "b", "c", "d", "e", None, "g"],
    }
)


class TestFloorAndCeil:
    def test_a_decimal_goes_down_or_up_to_a_whole_number_and_stays_a_decimal(self) -> None:
        floor = _values(NUMBERS, "floor([x])")
        ceil = _values(NUMBERS, "ceil([x])")
        assert floor[:6] == [-3.0, 2.0, -1.0, 0.0, 0.0, None] and math.isnan(floor[6])
        assert ceil[:6] == [-2.0, 3.0, 0.0, 1.0, 0.0, None] and math.isnan(ceil[6])
        assert _dtype(NUMBERS, "floor([x])") == pl.Float64 and _dtype(NUMBERS, "ceil([f32])") == pl.Float32

    def test_a_zero_never_carries_a_minus_sign(self) -> None:
        """ceil(-0.5) is 0, not -0: a minus sign on a zero reads as a debit in a written file."""
        for text in ("ceil([x])", "floor([x])"):
            zeros = [value for value in _values(NUMBERS, text)[:5] if value == 0]
            assert zeros and all(math.copysign(1.0, value) == 1.0 for value in zeros), text

    def test_a_whole_number_comes_back_as_it_is_in_its_own_type(self) -> None:
        assert _values(NUMBERS, "floor([n])") == [-3, 3, 0, 1, 0, None, 7]
        assert _values(NUMBERS, "ceil([small])") == [-3, 3, 0, 1, 0, None, 7]
        assert _dtype(NUMBERS, "ceil([small])") == pl.Int32

    def test_an_exact_decimal_stays_exact_with_its_places(self) -> None:
        assert _dtype(NUMBERS, "floor([money])") == pl.Decimal(10, 2)
        assert _values(NUMBERS, "floor([money])")[:3] == [Decimal("-2.00"), Decimal("1.00"), Decimal("-1.00")]
        assert _values(NUMBERS, "ceil([money])")[:3] == [Decimal("-1.00"), Decimal("2.00"), Decimal("0.00")]

    def test_null_gives_null_and_text_is_refused(self) -> None:
        assert _values(NUMBERS, "floor(null)") == [None]
        assert "floor()" in _refusal(NUMBERS, "floor([code])")
        assert "ceil()" in _refusal(NUMBERS, "ceil('1.5')")


class TestSign:
    def test_minus_one_zero_or_one_in_the_numbers_own_type(self) -> None:
        signs = _values(NUMBERS, "sign([x])")
        assert signs[:6] == [-1.0, 1.0, -1.0, 1.0, 0.0, None] and math.isnan(signs[6])
        assert math.copysign(1.0, signs[4]) == 1.0
        assert _values(NUMBERS, "sign([n])") == [-1, 1, 0, 1, 0, None, 1]
        assert _dtype(NUMBERS, "sign([small])") == pl.Int32

    def test_null_gives_null_and_text_is_refused(self) -> None:
        assert _values(NUMBERS, "sign(null)") == [None]
        assert "sign()" in _refusal(NUMBERS, "sign([code])")


class TestMod:
    PAIRS = pl.DataFrame(
        {
            "a": [7, -7, 7, -7, 0, None, 7, 6],
            "b": [3, 3, -3, -3, 3, 3, None, 3],
        }
    )

    def test_the_remainder_takes_the_sign_of_the_first_argument(self) -> None:
        """As SQL's MOD and Alteryx's Mod: mod(-7, 3) is -1. The engine's % would give 2."""
        assert _values(self.PAIRS, "mod([a], [b])") == [1, -1, 1, -1, 0, None, None, 0]
        assert _dtype(self.PAIRS, "mod([a], [b])") == pl.Int64

    def test_decimals_follow_the_same_rule(self) -> None:
        frame = pl.DataFrame({"a": [7.5, -7.5, 7.5, -7.5], "b": [2.0, 2.0, -2.0, -2.0]})
        assert _values(frame, "mod([a], [b])") == [1.5, -1.5, 1.5, -1.5]
        assert _values(self.PAIRS.head(2), "mod([a], 2.5)") == [2.0, -2.0]
        assert _dtype(self.PAIRS, "mod([a], 2.5)") == pl.Float64

    def test_an_unsigned_whole_number_meets_a_signed_one_as_a_whole_number(self) -> None:
        frame = pl.DataFrame({"a": pl.Series([7], dtype=pl.UInt64), "b": [-3]})
        assert _values(frame, "mod([a], [b])") == [1] and _dtype(frame, "mod([a], [b])") == pl.Int64

    def test_a_zero_divisor_stops_the_step_as_a_division_by_zero_does(self) -> None:
        frame = pl.DataFrame({"key": ["p", "q", "r"], "a": [7, 8, 9], "b": [3, 0, None]})
        with pytest.raises(DataError) as raised:
            _formula(frame, "mod([a], [b])")
        said = str(raised.value)
        assert "divides by zero" in said and "`[b]`" in said and "1 of 3 row(s)" in said and "'key': 'q'" in said
        assert _formula(frame.filter(pl.col("key") != "q"), "mod([a], [b])") == [1, None]

    @pytest.mark.parametrize("zero", ["0", "0.0", "-0.0"])
    def test_a_divisor_written_as_zero_is_refused_when_the_file_loads(self, zero: str) -> None:
        with pytest.raises(ExpressionError) as raised:
            compile_expr(f"mod([a], {zero})")
        assert "mod()" in str(raised.value) and "zero" in str(raised.value)

    def test_null_on_either_side_gives_null(self) -> None:
        assert _values(self.PAIRS.head(1), "mod(null, [b])") == [None]
        assert _values(self.PAIRS.head(1), "mod([a], null)") == [None]

    def test_an_exact_decimal_is_refused_with_what_to_do(self) -> None:
        said = _refusal(NUMBERS, "mod([money], 2)")
        assert "[money]" in said and "Float64" in said and "cast_columns" in said
        assert "Float64" in _refusal(NUMBERS, "mod([n], [money])")

    def test_text_is_refused(self) -> None:
        assert "mod()" in _refusal(NUMBERS, "mod([code], 2)")

    def test_the_most_negative_whole_number_keeps_its_own_sign_like_its_neighbour(self) -> None:
        """Its size does not fit its own type, so the engine's abs() wraps; the answer must not."""
        least = -(2**63)
        frame = pl.DataFrame(
            {
                "a": [least, least + 1, least, least, least, 5, -5, least],
                "b": [3, 3, -3, -1, 1, least, least, least],
            }
        )
        assert _values(frame, "mod([a], [b])") == [-2, -1, -2, 0, 0, 5, -5, 0]

    def test_the_most_negative_32_bit_whole_number_too(self) -> None:
        least = -(2**31)
        frame = pl.DataFrame({"a": [least, least + 1, least], "b": [3, 3, -1]}, schema={"a": pl.Int32, "b": pl.Int32})
        assert _values(frame, "mod([a], [b])") == [-2, -1, 0]
        assert _dtype(frame, "mod([a], [b])") == pl.Int32


class TestLeastAndGreatest:
    def test_nulls_are_skipped_and_only_all_nulls_give_null(self) -> None:
        frame = pl.DataFrame({"a": [3, None, None, 5], "b": [1, 4, None, 5], "c": [2, 6, None, None]})
        assert _values(frame, "least([a], [b], [c])") == [1, 4, None, 5]
        assert _values(frame, "greatest([a], [b], [c])") == [3, 6, None, 5]
        assert _values(frame, "greatest([a], null)") == [3, None, None, 5]

    def test_whole_numbers_and_decimals_together_give_a_decimal(self) -> None:
        frame = pl.DataFrame({"a": [3, 1], "b": [2.5, 1.5]})
        assert _values(frame, "least([a], [b])") == [2.5, 1.0]
        assert _dtype(frame, "greatest([a], [b])") == pl.Float64

    def test_text_and_dates_compare_as_comparisons_do(self) -> None:
        frame = pl.DataFrame(
            {"s": ["b", "B"], "t": ["a", "a"], "d": [date(2026, 1, 2), None], "e": [date(2026, 1, 1)] * 2}
        )
        assert _values(frame, "least([s], [t])") == ["a", "B"]
        assert _values(frame, "greatest([d], [e])") == [date(2026, 1, 2), date(2026, 1, 1)]

    def test_not_a_number_counts_as_larger_than_every_number(self) -> None:
        frame = pl.DataFrame({"a": [float("nan")], "b": [1.0]})
        assert _values(frame, "least([a], [b])") == [1.0]
        assert math.isnan(_values(frame, "greatest([a], [b])")[0])

    @pytest.mark.parametrize("function", ["least", "greatest"])
    def test_values_of_two_kinds_are_refused(self, function: str) -> None:
        assert f"{function}()" in _refusal(NUMBERS, f"{function}([x], [code])")
        assert "at least 2" in _refusal(NUMBERS, f"{function}([x])")

    @pytest.mark.parametrize("function", ["least", "greatest"])
    def test_timestamps_in_different_time_zones_are_refused(self, function: str) -> None:
        frame = pl.DataFrame({"a": [datetime(2026, 1, 1)], "b": [datetime(2026, 1, 1)]}).with_columns(
            pl.col("b").dt.replace_time_zone("UTC")
        )
        said = _refusal(frame, f"{function}([a], [b])")
        assert f"{function}()" in said and "UTC" in said and "cast_columns" in said
        assert _values(frame, f"{function}([b], [b])") == frame["b"].to_list()


class TestNullif:
    def test_null_where_the_two_are_equal_and_the_first_otherwise(self) -> None:
        frame = pl.DataFrame({"a": [0, 5, None, 0], "b": [0, 0, 0, None]})
        assert _values(frame, "nullif([a], [b])") == [None, 5, None, 0]
        assert _values(frame, "nullif([a], 0)") == [None, 5, None, None]
        assert _dtype(frame, "nullif([a], 0)") == pl.Int64

    def test_it_works_on_text_and_dates(self) -> None:
        frame = pl.DataFrame({"s": ["", "x", None], "d": [date(1900, 1, 1), date(2026, 1, 1), None]})
        assert _values(frame, "nullif([s], '')") == [None, "x", None]
        assert _values(frame, "nullif([d], date'1900-01-01')") == [None, date(2026, 1, 1), None]

    def test_two_kinds_are_refused(self) -> None:
        assert "nullif()" in _refusal(NUMBERS, "nullif([n], 'x')")

    def test_a_timestamp_against_a_date_is_refused_as_equals_refuses_it(self) -> None:
        frame = pl.DataFrame({"ts": [datetime(2026, 1, 1, 12)], "d": [date(2026, 1, 1)]})
        assert "midnight" in _refusal(frame, "nullif([ts], [d])")
        assert "midnight" in _refusal(frame, "nullif([d], [ts])")
